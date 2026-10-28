# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .torch_edge import dense_knn_matrix
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath

class MaxE(nn.Module):
    """
    
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MaxE, self).__init__()
        self.nn = BasicConv([in_channels*4, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
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
        return self.nn(x)


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
                
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='maxe', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'maxe':
            self.gconv = MaxE(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='maxe', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, is_print=False):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

        self.is_print = is_print

    def forward(self, x, relative_pos=None, tgi_guide=None):
        B, C, H, W = x.shape

        x_reshaped = x.reshape(x.shape[0], x.shape[1], -1, 1).contiguous()

        # 用于计算KNN的特征：如果有TGI引导，则concat
        if tgi_guide is not None:
            tgi_reshaped = tgi_guide.reshape(tgi_guide.shape[0], tgi_guide.shape[1], -1, 1).contiguous()
            x_for_knn = torch.cat([x_reshaped, tgi_reshaped], dim=1)
        else:
            x_for_knn = x_reshaped

        # y用于KNN计算，需要从x_for_knn得到
        y_for_knn = None
        y = None
        if self.r > 1:
            if tgi_guide is not None:
                x_concat = torch.cat([x, tgi_guide], dim=1)
                y_for_knn = F.avg_pool2d(x_concat, self.r, self.r)
                y_for_knn = y_for_knn.reshape(y_for_knn.shape[0], y_for_knn.shape[1], -1, 1).contiguous()
            else:
                y_for_knn = F.avg_pool2d(x, self.r, self.r)
                y_for_knn = y_for_knn.reshape(y_for_knn.shape[0], y_for_knn.shape[1], -1, 1).contiguous()

            # y用于聚合，只用原始x
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(y.shape[0], y.shape[1], -1, 1).contiguous()

        edge_index = self.dilated_knn_graph(x_for_knn, y_for_knn, relative_pos)

        # 聚合时只用原始ViG特征
        x_out = super(DyGraphConv2d, self).forward(x_reshaped, edge_index, y)

        return x_out.reshape(B, -1, H, W).contiguous()

import numpy as np
from scipy.spatial.distance import cosine

def calculate_graph_similarity_pairs(neighbor_tensor: torch.Tensor, output_path: str = "graph_similarity.txt"):
    """
    计算ViG网络中成对样本的图结构相似性
    
    参数说明：
    - neighbor_tensor: 输入Tensor，形状为 (2, B, C, K)，值为邻居的节点索引（整数）
      注意：C必须等于14×14=196（14×14网格的总节点数）
    - output_path: 输出txt文件路径，默认保存为 graph_similarity.txt
    """
    # 步骤1：处理第0维（假设是冗余维度，取第0维简化为 (B, C, K)）
    neighbor_tensor = neighbor_tensor[0]  # 简化后形状：(B, C, K)
    
    # 步骤2：获取核心维度参数
    B = neighbor_tensor.shape[0]  # 样本数
    C = neighbor_tensor.shape[1]  # 每个样本的节点数
    K = neighbor_tensor.shape[2]  # 每个节点的邻居数
    grid_size = 14
    
    # 验证C是否为14×14=196
    assert C == grid_size * grid_size, \
        f"节点数C必须等于14×14=196，当前C={C}！"
    
    # 步骤3：确保Tensor是整数类型
    if neighbor_tensor.dtype != torch.long and neighbor_tensor.dtype != torch.int:
        neighbor_tensor = neighbor_tensor.int()
    
    # 辅助函数：计算边重叠度（有向图）
    def calculate_edge_overlap(adj1, adj2):
        """计算两张图的边重叠度"""
        edges1 = set()
        edges2 = set()
        
        for node in range(C):
            for neighbor in adj1[node]:
                edges1.add((node, neighbor.item() if isinstance(neighbor, torch.Tensor) else neighbor))
            for neighbor in adj2[node]:
                edges2.add((node, neighbor.item() if isinstance(neighbor, torch.Tensor) else neighbor))
        
        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)
        
        return intersection / union if union > 0 else 1.0
    
    # 辅助函数：计算Jaccard相似系数（基于邻居集合）
    def calculate_jaccard_similarity(adj1, adj2):
        """计算两张图的Jaccard相似系数"""
        total_jaccard = 0.0
        
        for node in range(C):
            set1 = set(adj1[node].tolist())
            set2 = set(adj2[node].tolist())
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            if union == 0:
                node_jaccard = 1.0
            else:
                node_jaccard = intersection / union
            
            total_jaccard += node_jaccard
        
        return total_jaccard / C
    
    # 辅助函数：计算余弦相似度（基于邻接向量）
    def calculate_cosine_similarity(adj1, adj2):
        """计算两张图的余弦相似度"""
        adj_matrix1 = np.zeros((C, C))
        adj_matrix2 = np.zeros((C, C))
        
        for node in range(C):
            for neighbor in adj1[node]:
                adj_matrix1[node, neighbor] = 1
            for neighbor in adj2[node]:
                adj_matrix2[node, neighbor] = 1
        
        vec1 = adj_matrix1.flatten()
        vec2 = adj_matrix2.flatten()
        
        cos_sim = 1.0 - cosine(vec1, vec2)
        return (cos_sim + 1.0) / 2.0  # 归一化到[0, 1]
    
    # 步骤4：按B/2组队计算相似度
    num_pairs = B // 2
    pair_similarities = []
    
    with open(output_path, "w", encoding="utf-8") as f:
        # 写入分析头部
        f.write("=== ViG图结构相似性分析 ===\n\n")
        f.write(f"总样本数 B = {B}\n")
        f.write(f"配对数量 = {num_pairs} (第i个样本与第i+{num_pairs}个样本配对)\n")
        f.write(f"节点数 C = {C} (14×14网格)\n")
        f.write(f"每个节点的邻居数 K = {K}\n\n")
        f.write("=" * 70 + "\n\n")
        
        # 计算并记录每对样本的相似度
        for pair_idx in range(num_pairs):
            idx1 = pair_idx
            idx2 = pair_idx + num_pairs
            
            adj1 = neighbor_tensor[idx1]
            adj2 = neighbor_tensor[idx2]
            
            edge_overlap = calculate_edge_overlap(adj1, adj2)
            jaccard_sim = calculate_jaccard_similarity(adj1, adj2)
            cosine_sim = calculate_cosine_similarity(adj1, adj2)
            avg_sim = (edge_overlap + jaccard_sim + cosine_sim) / 3.0
            
            pair_similarities.append((edge_overlap, jaccard_sim, cosine_sim, avg_sim))
            
            # 写入当前配对结果
            f.write(f"配对 {pair_idx + 1}: 样本{idx1} ↔ 样本{idx2}\n")
            f.write(f"  边重叠度: {edge_overlap:.6f}\n")
            f.write(f"  Jaccard相似系数: {jaccard_sim:.6f}\n")
            f.write(f"  余弦相似度: {cosine_sim:.6f}\n")
            f.write(f"  综合相似度: {avg_sim:.6f}\n")
            f.write("-" * 50 + "\n\n")
        
        # 步骤5：计算统计信息
        if pair_similarities:
            # 提取各项指标
            edge_values = [s[0] for s in pair_similarities]
            jaccard_values = [s[1] for s in pair_similarities]
            cosine_values = [s[2] for s in pair_similarities]
            avg_values = [s[3] for s in pair_similarities]
            
            # 计算平均值
            avg_edge = np.mean(edge_values)
            avg_jaccard = np.mean(jaccard_values)
            avg_cosine = np.mean(cosine_values)
            avg_overall = np.mean(avg_values)
            
            # 计算标准差
            std_edge = np.std(edge_values)
            std_jaccard = np.std(jaccard_values)
            std_cosine = np.std(cosine_values)
            std_overall = np.std(avg_values)
            
            # 计算最大值和最小值
            max_edge = np.max(edge_values)
            min_edge = np.min(edge_values)
            max_jaccard = np.max(jaccard_values)
            min_jaccard = np.min(jaccard_values)
            max_cosine = np.max(cosine_values)
            min_cosine = np.min(cosine_values)
            max_overall = np.max(avg_values)
            min_overall = np.min(avg_values)
            
            # 写入统计信息
            f.write("=" * 70 + "\n")
            f.write("=== 相似度统计摘要 ===\n\n")
            
            f.write("1. 边重叠度统计:\n")
            f.write(f"   平均值: {avg_edge:.6f}, 标准差: {std_edge:.6f}\n")
            f.write(f"   最大值: {max_edge:.6f} (配对 {np.argmax(edge_values) + 1})\n")
            f.write(f"   最小值: {min_edge:.6f} (配对 {np.argmin(edge_values) + 1})\n\n")
            
            f.write("2. Jaccard相似系数统计:\n")
            f.write(f"   平均值: {avg_jaccard:.6f}, 标准差: {std_jaccard:.6f}\n")
            f.write(f"   最大值: {max_jaccard:.6f} (配对 {np.argmax(jaccard_values) + 1})\n")
            f.write(f"   最小值: {min_jaccard:.6f} (配对 {np.argmin(jaccard_values) + 1})\n\n")
            
            f.write("3. 余弦相似度统计:\n")
            f.write(f"   平均值: {avg_cosine:.6f}, 标准差: {std_cosine:.6f}\n")
            f.write(f"   最大值: {max_cosine:.6f} (配对 {np.argmax(cosine_values) + 1})\n")
            f.write(f"   最小值: {min_cosine:.6f} (配对 {np.argmin(cosine_values) + 1})\n\n")
            
            f.write("4. 综合相似度统计:\n")
            f.write(f"   平均值: {avg_overall:.6f}, 标准差: {std_overall:.6f}\n")
            f.write(f"   最大值: {max_overall:.6f} (配对 {np.argmax(avg_values) + 1})\n")
            f.write(f"   最小值: {min_overall:.6f} (配对 {np.argmin(avg_values) + 1})\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("=== 各指标相关性分析 ===\n\n")
            
            # 计算各指标间的相关性
            edge_jaccard_corr = np.corrcoef(edge_values, jaccard_values)[0, 1]
            edge_cosine_corr = np.corrcoef(edge_values, cosine_values)[0, 1]
            jaccard_cosine_corr = np.corrcoef(jaccard_values, cosine_values)[0, 1]
            
            f.write(f"边重叠度与Jaccard相似系数的相关性: {edge_jaccard_corr:.6f}\n")
            f.write(f"边重叠度与余弦相似度的相关性: {edge_cosine_corr:.6f}\n")
            f.write(f"Jaccard相似系数与余弦相似度的相关性: {jaccard_cosine_corr:.6f}\n")
    
    print(f"图相似性分析已完成，结果保存到：{output_path}")
    print(f"共分析了 {num_pairs} 对样本")
    print(f"平均相似度 - 边重叠度: {avg_edge:.4f}, Jaccard: {avg_jaccard:.4f}, 余弦: {avg_cosine:.4f}, 综合: {avg_overall:.4f}")
    
    return pair_similarities

class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='maxe', act='relu', norm=None, bias=True, stochastic=False,
                   epsilon=0.0, r=1, n=[192,256], drop_path=0.0, relative_pos=False, blocks=0, j=0, is_print=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.kernel_size = kernel_size

        # self.c_local = in_channels//2 if j<blocks//2 else in_channels//(2**(1+j-blocks//2))
        self.c_local = 0
        # if blocks == 2:
        #     self.c_local = 0
        self.c_graph = in_channels - self.c_local

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(self.c_graph, self.c_graph*2 , self.kernel_size, dilation, conv,
                            act, norm, bias, stochastic, epsilon, r, is_print)
        
        self.fc2 = nn.Sequential(
            nn.Conv2d(self.c_graph*2, self.c_graph, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.c_graph),
        )
    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(self.c_graph,
                int((n[0]*n[1])**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n[0]*n[1], int((self.n[0] - self.r) / self.r + 1)*int((self.n[1] - self.r) / self.r + 1)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

            # self.local_vertix = local_vertix(self.c_local,self.c_local)


    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n[0] * self.n[1]:
            return relative_pos
        else:
            N_reduced = int((H - self.r) / self.r + 1)*int((W - self.r) / self.r + 1)
            return F.interpolate(relative_pos.unsqueeze(0), size=(H*W, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x, tgi_guide=None):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)

        x = self.graph_conv(x, relative_pos, tgi_guide=tgi_guide)
        x = self.fc2(x)

        x = self.drop_path(x) + _tmp
        return x


class local_vertix(nn.Module):
    r"""
    输入: b,c,h,w
    输出: b,c,h*w

    进行local（周边8个点）的聚合，使用QKV机制，计算中不包含中心节点本身
    """
    def __init__(self, d, d_qk, window_size=[6,8]):
        super().__init__()
        self.d = d
        self.d_qk = d_qk
        self.window_size = window_size

        self.q_linear = nn.Linear(d,d_qk)
        self.k_linear = nn.Linear(d,d_qk)
        self.v_linear = nn.Linear(d,d_qk)

        self.norm1 = nn.BatchNorm2d(self.d)
        self.FFN = nn.Sequential(
            nn.Conv2d(self.d, self.d, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.d),
        )
        self.norm2 = nn.BatchNorm2d(self.d)

    def forward(self, x):
        B,C,H,W = x.shape

        Q = self.q_linear(torch.permute(x, (0, 2, 3, 1)))  # (B, H, W, C)

        neighbors = get_K(x)    # (B, H, W, 8, C)
        K = self.k_linear(neighbors)  # (B, H, W, 8, C)


        QK = torch.sum(Q.unsqueeze(-2) * K, dim=-1)  # (B, H, W, 8)

        # 计算权重 (softmax)
        weights = F.softmax(QK, dim=-1)  # (B, H, W, 8)

        # 周围点加权求和
        V_neigh = self.v_linear(neighbors)  # (B, H, W, 8, C)
        output = torch.sum(weights.unsqueeze(-1) * V_neigh, dim=-2)  # (B, H, W, C)
        output = output.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        x = self.norm1(output + x)
        output = self.FFN(x)
        output = self.norm2(output + x)
        return output


def get_K(x):
    r"""
    输入: b,c,h,w
    输出: b,c,h*w,8
    """
    B,C,H,W = x.shape
    x_padded = F.pad(x, (1, 1, 1, 1))  # (B, C, H+2, W+2)
    # Step 2: 提取邻域特征 (通过滑动窗口)
    # 定义相邻点的偏移量 (相对于中心点的 (0, 0))
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),  # 左上、正上、右上
        (0, -1),          (0, 1),   # 左、右
        (1, -1), (1, 0), (1, 1)     # 左下、正下、右下
    ]

    # 存储结果
    neighbors = []

    for dy, dx in offsets:
        # 提取每个方向的特征
        shifted = x_padded[:, :, 1+dy:H+1+dy, 1+dx:W+1+dx]  # 根据偏移量裁剪
        neighbors.append(shifted)

    # Step 3: 堆叠 8 个方向的结果
    output = torch.stack(neighbors, dim=4)  # (B, C, H, W) -> (B, C, H, W, 8)
    output = output.permute(0, 2, 3, 4, 1)  # (B, C, H, W, 8) -> (B, H, W, 8, C)
    return output


