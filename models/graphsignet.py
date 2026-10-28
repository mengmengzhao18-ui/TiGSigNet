"""
ContextCluster implementation
# --------------------------------------------------------
# Context Cluster -- Image as Set of Points, ICLR'23 Oral
# Licensed under The MIT License [see LICENSE for details]
# Written by Xu Ma (ma.xu1@northeastern.com)
# --------------------------------------------------------
"""
import os
import copy
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math

from collections import OrderedDict
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import to_2tuple
from einops import rearrange
import torch.nn.functional as F

from models.graphsignet_copy import Vig3

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'model_small': _cfg(crop_pct=0.9),
    'model_medium': _cfg(crop_pct=0.95),
}

def conv_bn_gelu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('gelu', nn.GELU()),
    ]))

class PointRecuder(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, input):
        x, tokens, total_codebook_loss = input
        x = self.proj(x)
        x = self.norm(x)
        return x, tokens, total_codebook_loss
    
class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Stem_G(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, in_dim=128, out_dim=768, act='gelu'):
        super().__init__()  
        self.convs = nn.Sequential(

        nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        )
        self.conv1 = conv_bn_gelu(in_dim, 32, 3, stride=2, pad=1)
        self.conv2 = conv_bn_gelu(in_dim, 32, 5, stride=2, pad=2)
        self.conv3 = conv_bn_gelu(in_dim, 32, 7, stride=2, pad=3)
        self.conv4 = nn.Conv2d(32*3, 64, kernel_size=1, stride=1)
        self.conv5 = conv_bn_gelu(64, 64, 3, stride=2, pad=1)
        self.conv6 = conv_bn_gelu(64, 64, 5, stride=2, pad=2)
        self.conv7 = conv_bn_gelu(64, 64, 7, stride=2, pad=3)
        self.conv8 = nn.Conv2d(64*3, out_dim, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.conv4(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))
        x = self.conv8(torch.cat([self.conv5(x), self.conv6(x), self.conv7(x)], dim=1))
        x = self.convs(x)
        return x

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


def pairwise_euclidean_dist(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise Euclidean distance matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: distance matrix [B,...,M,N]
    """
    # x1: [B,...,M,1,D]
    # x2: [B,...,1,N,D]
    x1_exp = x1.unsqueeze(-2)  # [B,...,M,1,D]
    x2_exp = x2.unsqueeze(-3)  # [B,...,1,N,D]

    # squared difference and sum
    dist = torch.sum((x1_exp - x2_exp) ** 2, dim=-1)  # [B,...,M,N]

    dist = torch.sqrt(dist + 1e-8)  # prevent sqrt(0)
    return dist

def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)

def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = pairwise_distance(x.detach())
        _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)

class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, x):

        #### normalize
        x = F.normalize(x, p=2.0, dim=1)
        ####
        edge_index = dense_knn_matrix(x, self.k * self.dilation)
        return edge_index

class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                m.append(nn.BatchNorm2d(channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(nn.GELU())
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature

class MaxE(nn.Module):
    """
    
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MaxE, self).__init__()
        self.nn = BasicConv([in_channels*4, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        b, c, n, _ = x.shape
        # if c==96:
        #     start_time = time.time()
        #     clusters, cluster_features = infomap_clustering(edge_index, x)
        #     end_time = time.time()
        #     print(f"Infomap clustering time: {end_time - start_time:.2f}s")
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        mean_j = torch.mean(x_j, dim=3, keepdim=True)
        std_j = torch.std(x_j, dim=3, keepdim=True)
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2), std_j.unsqueeze(2), mean_j.unsqueeze(2)], dim=2).reshape(b, 4 * c, n, _)
        # if c==96:
        #     end_time = time.time()
        #     print(f"MaxE forward time: {end_time - start_time:.2f}s")
        return self.nn(x)

class DyGraphConv2d(nn.Module):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='maxe', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__()
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        self.gnn = MaxE(in_channels, out_channels, act, norm, bias)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = x.reshape(x.shape[0], x.shape[1], -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x)
        x = self.gnn(x, edge_index)

        x = x.reshape(B, -1, H, W).contiguous()
        x = F.layer_norm(x, [x.shape[-1]])
        x = F.dropout(x, p=0.1, training=self.training)
        return x

class TokenEncoder(nn.Module):
    def __init__(self, in_dim, head_dim, heads, token_num=1):
        super().__init__()
        self.token_num = token_num
        self.heads = heads
        self.head_dim = head_dim
        out_dim = head_dim * heads

        self.conv1 = nn.Conv2d(in_dim, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(in_dim, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(in_dim, 32, 7, padding=3)
        self.fuse = nn.Conv2d(32 * 3, 64, 1)
        self.token_conv = nn.Conv2d(64, token_num * out_dim, 3, padding=1)
        self.norm = nn.LayerNorm(head_dim)

    def forward(self, x):  # x: [B, C, H, W]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)
        x = self.token_conv(x)
        B, D_T, H, W = x.shape
        x = x.view(B, self.token_num, self.heads, self.head_dim, H * W)
        x, _ = torch.max(x, dim=-1)  # token pool
        x = rearrange(x, "b t h c -> (b h) t c")
        x = self.norm(x)
        return x

class VanillaAttentionBlock(nn.Module):
    def __init__(self, dim, out_dim, fold_w=2, fold_h=2, heads=4, head_dim=24):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.fold_w = fold_w
        self.fold_h = fold_h

        inner_dim = heads * head_dim
        self.q_proj = nn.Linear(dim, inner_dim)
        self.kv_proj = nn.Linear(dim, 2 * inner_dim)
        self.out_proj = nn.Linear(inner_dim, inner_dim)   # ✅ 保持维度一致
        self.token_fuse = nn.Conv2d(inner_dim, out_dim, kernel_size=1)  # ✅ 输入通道为 inner_dim

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, tokens=None, _=0):
        B, C, H, W = x.shape
        patch_h = H // self.fold_h
        patch_w = W // self.fold_w

        x_blocks = rearrange(x, "b c (fh h) (fw w) -> (b fh fw) c h w", fh=self.fold_h, fw=self.fold_w)
        x_flat = rearrange(x_blocks, "b c h w -> b (h w) c")

        q = self.q_proj(x_flat)
        kv = self.kv_proj(x_flat)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> (b h) n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=self.heads)

        attn = torch.einsum("bnd,bmd->bnm", q, k) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bnm,bmd->bnd", attn, v)

        out = F.layer_norm(out, [self.head_dim])
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        out = self.out_proj(out)   # ✅ 输出仍是 inner_dim 维

        updated = rearrange(
            out,
            "b (ph pw) c -> b c ph pw",
            ph=patch_h, pw=patch_w
        )
        updated = rearrange(
            updated,
            "(b fh fw) c h w -> b c (fh h) (fw w)",
            fh=self.fold_h, fw=self.fold_w
        )

        updated = self.token_fuse(updated)

        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        output = alpha * x + beta * updated

        return output, None, 0

class TokenInteractionBlock(nn.Module):
    def __init__(self, dim, out_dim, fold_w=2, fold_h=2, heads=4, head_dim=24):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.fold_w = fold_w
        self.fold_h = fold_h

        self.token_encoder = TokenEncoder(dim, head_dim, heads)
        self.kv_proj = nn.Linear(dim, 2 * heads * head_dim)
        self.token_out_proj = nn.Linear(head_dim, head_dim)
        self.token_fuse = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, tokens=None, _=0):
        B, C, H, W = x.shape
        patch_h = H // self.fold_h
        patch_w = W // self.fold_w

        x_blocks = rearrange(x, "b c (fh h) (fw w) -> (b fh fw) c h w", fh=self.fold_h, fw=self.fold_w)
        tokens = self.token_encoder(x_blocks)  # [B_blocks * heads, T, D]

        x_flat = rearrange(x_blocks, "b c h w -> b (h w) c")
        kv = self.kv_proj(x_flat)
        kv = rearrange(kv, "b n (kv h d) -> kv b h n d", kv=2, h=self.heads)
        k, v = kv[0], kv[1]
        k = rearrange(k, "b h n d -> (b h) n d")
        v = rearrange(v, "b h n d -> (b h) n d")

        attn = torch.einsum("btd,bnd->btn", tokens, k) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        token_out = torch.einsum("btn,bnd->btd", attn, v)
        token_out = F.layer_norm(token_out, [self.head_dim])
        token_out = self.token_out_proj(token_out)

        reverse_attn = attn.transpose(1, 2)
        updated = torch.einsum("bnt,btd->bnd", reverse_attn, token_out)

        updated = rearrange(updated, "(b fh fw h) (ph pw) c -> b (h c) (fh ph) (fw pw)", fh=self.fold_h, fw=self.fold_h,
                            h=self.heads, ph=patch_h, pw=patch_w)

        updated = self.token_fuse(updated)
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        out = alpha * x + beta * updated
        return out, token_out, 0


class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x


class ClusterBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.GroupNorm,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False, block_idx=0):

        super().__init__()
        if norm_layer == nn.GroupNorm:
            self.norm1 = norm_layer(1, dim)  # GroupNorm(num_groups=1, num_channels=dim)
            self.norm2 = norm_layer(1, dim)
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
        self.token_mixer = TokenInteractionBlock(dim, dim, fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim)
        # self.token_mixer = VanillaAttentionBlock(dim, dim, fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale_1 = self.layer_scale_2 = 1.0

        self.block_idx = block_idx
        self.fold_w = fold_w

    def forward(self, input):
        x, tokens, _ = input
        B, C, H, W = x.shape

        if self.block_idx % 2:
            shift = H // self.fold_w // 2
            x = torch.roll(x, shifts=(-shift, -shift), dims=(2, 3))

        x_out, tokens, _ = self.token_mixer(self.norm1(x), tokens)

        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x_out)
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))

        if self.block_idx % 2:
            x = torch.roll(x, shifts=(shift, shift), dims=(2, 3))

        return x, tokens, 0

def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.1, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * ( block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(ClusterBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
            heads=heads, head_dim=head_dim, return_center=False, block_idx = block_idx
        ))
    blocks = nn.Sequential(*blocks)

    return blocks


class ContextCluster(nn.Module):
    """
    ContextCluster, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, the embedding dims, mlp ratios
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained:
        for mmdetection and mmsegmentation to load pretrained weights
    """

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 output_dim=384,
                 in_patch_size=4, in_stride=4, in_pad=0,
                 down_patch_size=2, down_stride=2, down_pad=0,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 # the parameters for context-cluster
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1],
                 heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32],
                 **kwargs):

        super().__init__()

        self.heads = heads
        self.fold_w = fold_w    
        self.fold_h = fold_h

        self.output_dim = output_dim

        self.patch_embed = Stem_G(in_dim=1, out_dim=embed_dims[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dims[0], 224//4, 224//4))
        # self.patch_embed = PointRecuder(
        #     patch_size=in_patch_size, stride=in_stride, padding=in_pad,
        #     in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network1 = []
        i = 0
        stage = basic_blocks(embed_dims[i], i, layers,
                                mlp_ratio=mlp_ratios[i],
                                act_layer=act_layer, norm_layer=norm_layer,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                                fold_w=fold_w[i], fold_h=fold_h[i], heads=heads[i], head_dim=head_dim[i],
                                return_center=False
                                )
        network1.append(stage)
        if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
            # downsampling between two stages
            network1.append(
                # Downsample(in_dim=embed_dims[i], out_dim=embed_dims[i + 1])
                PointRecuder(
                    patch_size=down_patch_size, stride=down_stride,
                    padding=down_pad,
                    in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                )
            )
        self.network1 = nn.ModuleList(network1)

        network2 = []
        i = 1
        stage = basic_blocks(embed_dims[i], i, layers,
                                mlp_ratio=mlp_ratios[i],
                                act_layer=act_layer, norm_layer=norm_layer,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                                fold_w=fold_w[i], fold_h=fold_h[i], heads=heads[i], head_dim=head_dim[i],
                                return_center=False
                                )
        network2.append(stage)
        if len(embed_dims)>2:
            network2.append(
                # Downsample(in_dim=embed_dims[i], out_dim=embed_dims[i + 1])
                PointRecuder(
                    patch_size=down_patch_size, stride=down_stride,
                    padding=down_pad,
                    in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                )
            )    
        self.network2 = nn.ModuleList(network2)

        if len(embed_dims)>2:
            network3 = []
            i = 2
            stage = basic_blocks(embed_dims[i], i, layers,
                                    mlp_ratio=mlp_ratios[i],
                                    act_layer=act_layer, norm_layer=norm_layer,
                                    drop_rate=drop_rate,
                                    drop_path_rate=drop_path_rate,
                                    use_layer_scale=use_layer_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                                    fold_w=fold_w[i], fold_h=fold_h[i], heads=heads[i], head_dim=head_dim[i],
                                    return_center=False
                                    )
            network3.append(stage)

            self.network3 = nn.ModuleList(network3)

        self.tokens_conv1 = nn.Sequential(nn.Conv2d(head_dim[0], head_dim[1] * 2, kernel_size=2,stride=2, bias=True),
                                              nn.BatchNorm2d(head_dim[1] * 2),
                                              act_layer())
        if len(embed_dims)>2:
            self.tokens_conv2 = nn.Sequential(nn.Conv2d(head_dim[1]*2, head_dim[2] * 2, kernel_size=2,stride=2, bias=True),
                                              nn.BatchNorm2d(head_dim[2] * 2),
                                              act_layer())
            self.tokens_conv3 = nn.Sequential(nn.Conv1d(head_dim[-1]*2, head_dim[-1], kernel_size=1),
                                                )
        self.prediction = Seq(nn.Conv2d(embed_dims[-1], 512, 1, bias=True),
                              nn.BatchNorm2d(512),
                              act_layer(),
                              nn.Dropout(drop_rate),
                              nn.Conv2d(512, self.output_dim, 1, bias=True))

        # Token Pooling（结构化聚合）
        self.token_pool_proj = nn.Linear(64, 96)  # [B, 64, L] -> [B, 96, L]
        encoder_layer = TransformerEncoderLayer(d_model=96, nhead=4, dim_feedforward=192, dropout=0.1, activation="gelu", batch_first=True)
        self.token_transformer = TransformerEncoder(encoder_layer, num_layers=2)

        self.token_attn_pool = nn.Sequential(
            nn.Linear(96, 1),   # [B, L, 96] -> [B, L, 1]
            nn.Softmax(dim=1)   # softmax over tokens
        )
        # self.token_cls_head = nn.Sequential(
        #     nn.LayerNorm(96),
        #     nn.Linear(96, 96)
        # )

    def aggregate_global_features(self, x_updated, tokens_out):
        x_global = F.adaptive_avg_pool2d(x_updated, 1).squeeze(-1).squeeze(-1)
        tokens_out = rearrange(tokens_out, "(b e f1 f2) t c -> b e f1 f2 t c", e=self.heads[-1], f1=self.fold_w[-1], f2=self.fold_h[-1])
        token_global = tokens_out.mean(dim=[1, 2, 3, 4])
        out = torch.cat([x_global, token_global], dim=1)
        return out
    
    def forward_embeddings(self, x):
        _, c, img_w, img_h = x.shape
        x = self.patch_embed(x) + self.pos_embed
        return x
    
    def forward(self, x):
        x = self.forward_embeddings(x)

        

        b, _, _, _ = x.shape
        # tokens1 = self.tokens1.repeat(b * self.fold_h[0] * self.fold_w[0], 1, 1)  # [4096,6,64] 
        # tmp_tokens1 = tokens1
        tokens1 = None
        tokens2 = None
        tokens3 = None
        total_codebook_loss = 0
        for idx, block in enumerate(self.network1):
            x, tokens1, total_codebook_loss = block([x, tokens1, total_codebook_loss])
            # if idx == len(self.network1) - 2:
            #     save_color_heatmap_and_patched_images(x, 'before_transformer')
            if idx == len(self.network1) - 2:
                x_stage1 = x  # 保存stage1在PointReducer之前的特征

        tokens1 = rearrange(tokens1, "(b f1 f2 h) t c -> (b t h) c f1 f2", f1=self.fold_w[0], f2=self.fold_h[0], b=b, h=self.heads[0])
        tokens2 = self.tokens_conv1(tokens1)
        tokens2 = tokens2.view((tokens2.shape[0],2,-1,self.fold_w[1],self.fold_h[1]))
        tokens2 = rearrange(tokens2, "(b t h) mul c f1 f2 -> (b f1 f2 h mul) t c", f1=self.fold_w[1], f2=self.fold_h[1], b=b, t=1, mul=2)

        tmp_tokens2 = tokens2
        for idx, block in enumerate(self.network2):
            x, tokens2, total_codebook_loss = block([x, tokens2, total_codebook_loss])
            if idx == len(self.network2) - 2:
                x_out = x

        x_stage2 = x_out  # 保存stage2的特征

        tokens2 = torch.cat([tokens2, tmp_tokens2], dim=2)   
        tokens2 = rearrange(tokens2, "(b f1 f2 h) t c -> (b t h) c f1 f2", f1=self.fold_w[1], f2=self.fold_h[1], b=b, h=self.heads[1])
        tokens3 = self.tokens_conv2(tokens2)
        tokens3 = tokens3.view((tokens3.shape[0],2,-1,self.fold_w[2],self.fold_h[2]))
        tokens3 = rearrange(tokens3, "(b t h) mul c f1 f2 -> (b f1 f2 h mul) t c", f1=self.fold_w[2], f2=self.fold_h[2], b=b, t=1, mul=2)
        tmp_tokens3 = tokens3

        # tmp_tokens2 = torch.cat([tmp_tokens2], dim=2)   
        # tmp_tokens2 = rearrange(tmp_tokens2, "(b f1 f2 h) t c -> (b t h) c f1 f2", f1=self.fold_w[1], f2=self.fold_h[1], b=b, h=self.heads[1])
        # tmp_tokens2 = self.tokens_conv2(tmp_tokens2)
        # tmp_tokens2 = tmp_tokens2.view((tmp_tokens2.shape[0],2,-1,self.fold_w[2],self.fold_h[2]))
        # tmp_tokens2 = rearrange(tmp_tokens2, "(b t h) mul c f1 f2 -> (b f1 f2 h mul) t c", f1=self.fold_w[2], f2=self.fold_h[2], b=b, t=1, mul=2)
 
        
        for idx, block in enumerate(self.network3):
            x, tokens3, total_codebook_loss = block([x, tokens3, total_codebook_loss])

        x_stage3 = x  # 保存stage3的特征

        tokens3 = torch.cat([tokens3, tmp_tokens3], dim=2)
        tokens3 = rearrange(tokens3, "(b f1 f2 h) t c -> b c (f1 f2 h t)", b=b, f1=self.fold_w[2], f2=self.fold_h[2], t=1, h=self.heads[2])
        tokens3 = self.tokens_conv3(tokens3)

        # tokens3 shape: [B, C, L]
        tokens3 = self.token_pool_proj(tokens3.permute(0, 2, 1))  # [B, L, C] → [B, L, 96]
        tokens3 = self.token_transformer(tokens3)  # [B, L, 96]
        attn_weights = self.token_attn_pool(tokens3)  # [B, L, 1]
        token_embed = torch.sum(tokens3 * attn_weights, dim=1)  # [B, 96]

        # 返回三个stage的特征用于引导ViG
        return [x_stage1, x_stage2, x_stage3], token_embed, total_codebook_loss


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt # 🚀 引入 Matplotlib

def save_color_heatmap_and_patched_images(
    tensor: torch.Tensor,
    original_dir: str = "original_color_heatmaps",
    patched_dir: str = "patched_color_heatmaps_with_gap",
    gap: int = 2,              # 块之间的空距（像素）
    normalize: bool = True,    # 是否自动归一化到[0,1]再映射颜色
    pad_len: int = 4,          # 文件名序号补零位数
    colormap_name: str = 'viridis' # 🆒 默认使用 Viridis 配色，也可以试试 'jet' 或 'magma'
):
    """
    处理 B, C, 56, 56 的Tensor，先沿C轴取平均降为单通道值，
    然后使用 Matplotlib Colormap 映射成彩色热力图并保存。
    
    张量形状要求: (B, C, 56, 56)
    """
    # --- 1. 数据预处理与形状验证 ---
    B, C, H, W = tensor.shape
    GRID_SIZE = 14
    ORIGINAL_SIZE = 56
    
    assert H == ORIGINAL_SIZE and W == ORIGINAL_SIZE, \
        f"Tensor空间尺寸必须是 ({ORIGINAL_SIZE},{ORIGINAL_SIZE})，当前是 ({H}, {W})"

    # 降维打击：沿通道方向取平均 (B, C, 56, 56) -> (B, 56, 56)
    avg_img_tensor = tensor.mean(dim=1).cpu() # 直接转到CPU，后续处理

    # 获取颜色映射表（Colormap）
    try:
        cmap = plt.cm.get_cmap(colormap_name)
    except AttributeError:
        # 兼容旧版本 matplotlib
        cmap = plt.get_cmap(colormap_name) 

    # --- 2. 创建保存文件夹 ---
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(patched_dir, exist_ok=True)

    # --- 3. 遍历每个样本处理 ---
    for idx in range(B):
        img = avg_img_tensor[idx] # 形状：(56, 56)
        
        # ---------------------- 步骤 A：归一化到 [0, 1] ----------------------
        if normalize:
            # 归一化到 [0, 1]
            img_min = img.min()
            img_max = img.max()
            if img_max == img_min:
                 # 全零/全同值时，直接设为0.0
                 img_norm = np.zeros((ORIGINAL_SIZE, ORIGINAL_SIZE), dtype=np.float32)
            else:
                 img_norm = ((img - img_min) / (img_max - img_min)).numpy()
        else:
            # 如果不需要自动归一化，确保值在 [0, 1] 范围内
            img_norm = img.numpy() 

        # ---------------------- 步骤 B：应用 Colormap 映射成彩色图 ----------------------
        # Colormap 会将 [0, 1] 的浮点数映射为 (H, W, 4) 的 RGBA 数组
        # 值范围已经是 [0, 255] 的 uint8 格式
        color_img_np = (cmap(img_norm) * 255).astype(np.uint8) 
        
        # 为了兼容保存 PNG，我们只取 RGB 三通道 (H, W, 3)
        rgb_img_np = color_img_np[..., :3] 

        # 保存原始彩色热力图
        original_path = os.path.join(original_dir, f"{idx:0{pad_len}d}.png")
        Image.fromarray(rgb_img_np).save(original_path)

        # ---------------------- 步骤 C：生成 14x14 彩色切块+空距拼接图 ----------------------
        patch_size = ORIGINAL_SIZE // GRID_SIZE  # 56 / 14 = 4 → (4, 4)
        num_patches = GRID_SIZE

        # 计算拼接图的总尺寸，注意画布现在是 RGB 三通道 (total_size, total_size, 3)
        total_size = num_patches * patch_size + (num_patches - 1) * gap
        # 创建白色背景画布 (255, 255, 255)
        patched_canvas = np.ones((total_size, total_size, 3), dtype=np.uint8) * 255

        # 遍历切块，使用彩色图进行切分
        for i in range(num_patches):  # 行
            for j in range(num_patches):  # 列
                # 1. 原始图坐标
                y_start, y_end = i * patch_size, (i + 1) * patch_size
                x_start, x_end = j * patch_size, (j + 1) * patch_size
                
                # 2. 提取当前彩色块 (4, 4, 3)
                patch = rgb_img_np[y_start:y_end, x_start:x_end]

                # 3. 画布坐标
                canvas_y_start = i * (patch_size + gap)
                canvas_y_end = canvas_y_start + patch_size
                canvas_x_start = j * (patch_size + gap)
                canvas_x_end = canvas_x_start + patch_size

                # 4. 将彩色块放入画布
                patched_canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = patch

        # 保存拼接图
        patched_path = os.path.join(patched_dir, f"{idx:0{pad_len}d}.png")
        Image.fromarray(patched_canvas).save(patched_path)

    print(f"😎 彩色热力图处理完成！已将 {B} 个 (C, 56, 56) Tensor 转换为 {colormap_name.upper()} 热力图并切块。")
    print(f"原始热力图：{os.path.abspath(original_dir)}（彩色 56x56）")
    print(f"切块拼接图：{os.path.abspath(patched_dir)}（彩色 4x4 块，空距：{gap}）")

def coc_base_dim64(pretrained=False, **kwargs):
    layers = [2, 2, 2]
    norm_layer = GroupNorm
    embed_dims = [96, 256, 384]
    mlp_ratios = [4, 4, 4]
    downsamples = [True, True, True]
    proposal_w = [2, 2, 2]
    proposal_h = [2, 2, 2]
    fold_w = [8, 4, 2]
    fold_h = [8, 4, 2]
    heads = [4, 8, 16]# 12,12,16
    head_dim = [64, 64, 64]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        drop_rate=0.1, drop_path_rate=0.0,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model

class Mix_Model(nn.Module):
    def __init__(self):
        super(Mix_Model, self).__init__()
        self.cluster = coc_base_dim64()
        self.vig = Vig3()

        self.conv1 = conv_bn_relu(1, 128, 3, stride=1, pad=1)
        self.conv2 = conv_bn_relu(128, 64, 3, stride=1, pad=1)
        self.conv3 = conv_bn_relu(64, 32, 3, stride=1, pad=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + residual

        x_2, token, total_codebook_loss = self.cluster(x)
        x2 = self.vig(x, tgi_guide=x_2)  # 传入TGI特征引导

        return torch.cat([token, x2], dim=1)
    
def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))


def linear_bn_relu(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))

class SigNetHaffman(nn.Module):
    """ SigNet model, from https://arxiv.org/abs/1705.05787
    """

    def __init__(self, weight=None):
        super(SigNetHaffman, self).__init__()

        self.feature_space_size = 768

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_relu(1, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', conv_bn_relu(96, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', conv_bn_relu(256, 384, 3, pad=1)),
            ('conv4', conv_bn_relu(384, 384, 3, pad=1)),
            ('conv5', conv_bn_relu(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_relu(256 * 5 * 5, 768)),
            ('fc2', linear_bn_relu(self.feature_space_size, self.feature_space_size)),
        ]))

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        x = x.view(x.shape[0], 256 * 5 * 5)
        x = self.fc_layers(x)
        return x


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))

if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    model = coc_base_dim64()
    out = model(input)
    print(model)
    print(out.shape)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {:.2f}M".format(n_parameters/1024**2))



