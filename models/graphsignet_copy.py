import os
import math
import random
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
from torch.nn import Sequential as Seq
import numpy as np
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from models.gcn_lib_copy import Grapher, act_layer


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def spd_conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(OrderedDict([
        ('spd', space_to_depth()),
        ('conv', Conv(in_channels, out_channels, k=1, s=1)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))


def conv_bn_gelu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('gelu', nn.GELU()),
    ]))


def linear_bn_relu(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

default_cfgs = {
    'vig_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='gelu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)

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

class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        """

        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        
    def forward(self, x):  # [b,c,w,h]
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h) # 这是每个点的特征
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h  这是用于计算相似性的聚类中心特征

        value_centers = self.centers_proposal(value)    # [b,c,C_W,C_H] 这是聚类中心的特征

        value_centers = rearrange(value_centers, 'b c w h -> b (w h) c')  # [b,C_W,C_H,c] 这是聚类中心的特征
        
        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]    # 这是每个点的特征

        
        # aggregate step, out shape [B,M,D]
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                    sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            # dispatch step, return to each point in a cluster
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out


class Stem_G(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=128, out_dim=768, act='gelu'):
        super().__init__()  
        self.convs = nn.Sequential(

        nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        )
        self.conv1 = conv_bn_gelu(1, 32, 3, stride=2, pad=1)
        self.conv2 = conv_bn_gelu(1, 32, 5, stride=2, pad=2)
        self.conv3 = conv_bn_gelu(1, 32, 7, stride=2, pad=3)
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
    
class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),   
        )
        
    def forward(self, x):
        return self.stem(x)

class Conv_Downsample(nn.Module):
    """ 
    3和5两个尺度进行卷积，然后拼接过linear层
    """
    def __init__(self, in_dim=128, out_dim=768, mid_dim=32, act='gelu'):
        super().__init__()  
        self.convs = nn.Sequential(
        nn.Conv2d(out_dim, out_dim, 1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
        )
        self.conv1 = conv_bn_gelu(in_dim, mid_dim, 3, stride=2, pad=1)
        self.conv2 = conv_bn_gelu(in_dim, mid_dim, 5, stride=2, pad=2)
        self.conv4 = nn.Conv2d(mid_dim*2, out_dim, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.conv4(torch.cat([self.conv1(x), self.conv2(x)], dim=1))
        x = self.convs(x)
        return x

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

class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        drop_path = opt.drop_path
        
        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [2,1,1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        # num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        # max_dilation = 49 // max(num_knn)
        max_dilation = 49 // max(k)
        c_in = opt.c_in
        H = 56
        W = 56

        self.stem = Stem(c_in, channels[0])

        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], H, W))  
        HW = H*W

        self.backbone1 = nn.ModuleList([])
        idx = 0
        i = 0
        for j in range(blocks[i]):
            self.backbone1 += [
                Seq(Grapher(channels[i], k[i], min(idx // 4 + 1, max_dilation), conv, act, norm,# num_knn[idx]
                                bias, stochastic, epsilon, reduce_ratios[i], n=[H,W], drop_path=dpr[idx],
                                relative_pos=True, blocks=blocks[i], j=j),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        )]
            idx += 1
        self.backbone1 = Seq(*self.backbone1)

        W = int((W-1) // 2)+1
        H = int((H-1) // 2)+1

        self.backbone2 = nn.ModuleList([])
        idx = 0
        i = 1
        self.backbone2 += [Downsample(channels[i-1], channels[i])]
        for j in range(blocks[i]):
            self.backbone2 += [
                Seq(Grapher(channels[i], k[i], min(idx // 4 + 1, max_dilation), conv, act, norm,# num_knn[idx]
                                bias, stochastic, epsilon, reduce_ratios[i], n=[H,W], drop_path=dpr[idx],
                                relative_pos=True, blocks=blocks[i], j=j),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        )]
            idx += 1
        self.backbone2 = Seq(*self.backbone2)

        W = int((W-1) // 2)+1
        H = int((H-1) // 2)+1

        self.backbone3 = nn.ModuleList([])
        idx = 0
        i = 2
        self.backbone3 += [Downsample(channels[i-1], channels[i])]
        for j in range(blocks[i]):
            self.backbone3 += [
                Seq(Grapher(channels[i], k[i], min(idx // 4 + 1, max_dilation), conv, act, norm,# num_knn[idx]
                                bias, stochastic, epsilon, reduce_ratios[i], n=[H,W], drop_path=dpr[idx],
                                relative_pos=True, blocks=blocks[i], j=j, is_print=(i==2 and j==1)),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        )]
            idx += 1
        self.backbone3 = Seq(*self.backbone3)
        
        # self.attn_pool = AttnPool(channels[-1])

        self.prediction = Seq(nn.Conv2d(channels[-1], 512, 1, bias=True),
                              nn.BatchNorm2d(512),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(512, opt.out_channel, 1, bias=True))
        
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs, tgi_guide=None):
        x = self.stem(inputs) + self.pos_embed

        # Stage 1: 所有block都是Seq(Grapher, FFN)
        for block in self.backbone1:
            if tgi_guide is not None:
                x = block[0](x, tgi_guide=tgi_guide[0])  # Grapher
                x = block[1](x)  # FFN
            else:
                x = block(x)

        # Stage 2: 第一个是Downsample，其余是Seq(Grapher, FFN)
        for i, block in enumerate(self.backbone2):
            if i == 0:
                x = block(x)  # Downsample
            else:
                if tgi_guide is not None:
                    x = block[0](x, tgi_guide=tgi_guide[1])
                    x = block[1](x)
                else:
                    x = block(x)

        # Stage 3: 第一个是Downsample，其余是Seq(Grapher, FFN)
        for i, block in enumerate(self.backbone3):
            if i == 0:
                x = block(x)  # Downsample
            else:
                if tgi_guide is not None:
                    x = block[0](x, tgi_guide=tgi_guide[2])
                    x = block[1](x)
                else:
                    x = block(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = self.prediction(x).squeeze(-1).squeeze(-1)
        return x

class Vig3(torch.nn.Module):
    def __init__(self):
        super(Vig3, self).__init__()

        self.vig = Swin_ViG(c_in=1, out_channel=384, channels=[96,256,384], drop_path_rate=0.0)

    def forward(self, inputs, tgi_guide=None):
        """进行原分支，可选TGI引导"""

        x_vig = self.vig(inputs, tgi_guide=tgi_guide)

        return x_vig
    
def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))

@register_model
def pvig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, out_channel=1024, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,6,2] # number of basic blocks in the backbone
            self.channels = [48, 96, 240, 384] # number of channels of deep features
            self.out_channel = out_channel # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    # model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,6,2] # number of basic blocks in the backbone
            self.channels = [80, 160, 400, 640] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    # model.default_cfg = default_cfgs['vig_224_gelu']
    return model


def pvig_m_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,16,2] # number of basic blocks in the backbone
            self.channels = [96, 192, 384, 768] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    # model.default_cfg = default_cfgs['vig_224_gelu']
    return model


def pvig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, out_channel=512, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.1 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,12,2] # number of basic blocks in the backbone
            self.channels = [64, 128, 256, 512] # number of channels of deep features
            self.out_channel = out_channel # Dimension of out_channels

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    # model.default_cfg = default_cfgs['vig_b_224_gelu']
    return model

def Swin_ViG (pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, c_in=1, out_channel=512, channels=[128], drop_path_rate=0.0, **kwargs):
            self.k = [8,4,2] # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.1 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,2] # number of basic blocks in the backbone
            self.channels = channels # number of channels of deep features
            self.out_channel = out_channel # Dimension of out_channels
            self.c_in = c_in

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    # model.default_cfg = default_cfgs['vig_b_224_gelu']
    return model

def Cluster_ViG (pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, c_in=256, out_channel=384, channels=[256,384], drop_path_rate=0.0, **kwargs):
            self.k = [4,2] # neighbor num (default:9)
            self.conv = 'maxe' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.1 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2] # number of basic blocks in the backbone
            self.channels = channels # number of channels of deep features
            self.out_channel = out_channel # Dimension of out_channels
            self.c_in = c_in

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    # model.default_cfg = default_cfgs['vig_b_224_gelu']
    return model




class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x
