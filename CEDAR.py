from runtime_paths import data_path, runtime_batch_size, runtime_workers
import os
import random
import itertools
from PIL import ImageOps, Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


# 路径配置
G_PATH = data_path('CEDAR', 'T_c')
F_PATH = data_path('CEDAR', 'F_c')

# 用户划分
# train_users = list(range(1, 51))
# test_users = list(range(51, 56))
random.seed(42)

test_users = [51, 40, 48, 16, 11]
train_users = [user for user in list(range(1, 56)) if user not in test_users]

# train_users = random.sample(list(range(1, 56)), 50)
# test_users = [user for user in list(range(1, 56)) if user not in train_users]

# 文件名格式：Forgeries_user_index.png，如 forgeries_1_1.png
def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split('_')  # 使用 '_' 分割
    uid = parts[1]  # 第二个部分是 uid
    return int(uid)

# 加载文件并按用户组织
def collect_signatures(g_path, f_path):
    user_data = defaultdict(lambda: {'genuine': [], 'forged': []})
    for fname in os.listdir(g_path):
        uid = parse_filename(fname)
        user_data[uid]['genuine'].append(os.path.join(g_path, fname))
    for fname in os.listdir(f_path):
        uid = parse_filename(fname)
        user_data[uid]['forged'].append(os.path.join(f_path, fname))
    return user_data

user_data = collect_signatures(G_PATH, F_PATH)

# 生成样本对
def generate_pairs(users):
    genuine_pairs = []
    fake_pairs_all = []

    for uid in users:
        genuine = user_data[uid]['genuine']
        forged = user_data[uid]['forged']
        if len(genuine) < 2 or len(forged) == 0:
            continue
        # 正样本：所有真签名的组合
        genuine_pairs.extend(itertools.combinations(genuine, 2))
        # 负样本：所有真伪配对
        fake_pairs_all.extend(itertools.product(genuine, forged))

    # 平衡负样本数量
    fake_pairs = random.sample(fake_pairs_all, len(genuine_pairs)) if len(fake_pairs_all) > len(genuine_pairs) else fake_pairs_all

    return genuine_pairs, fake_pairs

# Dataset
class SignaturePairDataset(Dataset):
    def __init__(self, genuine_pairs, fake_pairs, transform=None):
        self.pairs = genuine_pairs + fake_pairs
        self.labels = [1] * len(genuine_pairs) + [0] * len(fake_pairs)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2 = self.pairs[idx]
        label = self.labels[idx]

        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    ImageOps.invert,
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    ImageOps.invert,
    transforms.ToTensor()
])

# train_transform = transforms.Compose([
#     # 1. 基础预处理
#     transforms.Grayscale(num_output_channels=1),  # 确保灰度
#     ImageOps.invert,
    
#     # 2. 几何变形 (三选一)
#     transforms.RandomChoice([
#         transforms.ElasticTransform(alpha=15.0, sigma=3.0),  # 更温和的弹性变形
#         transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
#         transforms.RandomAffine(degrees=5.0, translate=(0.1, 0.1), scale=(0.9, 1.1))
#     ]),
    
#     # 3. 裁剪和外观增强
#     transforms.RandomCrop(224, padding=10, padding_mode='edge'),  # 边缘填充防止裁掉特征
#     transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.3),
    
#     # 4. 标准化
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度标准化
# ])

# test_transform = transforms.Compose([
#     # 1. 基础预处理（与训练集一致）
#     transforms.Resize([224, 224]),  # 直接缩放到目标尺寸（无需RandomCrop）
#     ImageOps.invert,
#     transforms.Grayscale(num_output_channels=1),
    
#     # 2. 仅包含确定性转换
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])  # 必须与训练集相同
# ])

# 构建数据集和加载器
train_g, train_f = generate_pairs(train_users)
test_g, test_f = generate_pairs(test_users)

train_dataset = SignaturePairDataset(train_g, train_f, transform=train_transform)
test_dataset = SignaturePairDataset(test_g, test_f, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=runtime_batch_size(16), shuffle=True, num_workers=runtime_workers(8), pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=runtime_batch_size(16), shuffle=True, num_workers=runtime_workers(8), pin_memory=True)
