from runtime_paths import data_path, runtime_batch_size, runtime_workers
import numpy as np
import torch
import itertools
from PIL import ImageOps, Image
from torch.utils.data import Dataset, DataLoader
import random

# 加载数据集
with np.load(data_path('GPDS_1000_256X256.npz'), allow_pickle=True) as data:
    x, y, yforg = data['x'], data['y'], data['yforg']

# 数据维度
# x: (N, 1, H, W)
# y: (N,) 用户编号
# yforg: (N,) 签名真伪，0表示真，1表示伪

# 获取所有用户编号
unique_users = np.unique(y)


# 随机选取600个用户作为训练集，200验证集，200测试集
random.seed(0)  # 设置随机种子以便复现
# all_users = np.array([int(user) for user in range(0, 300)])
all_users = unique_users
train_users = random.sample(list(all_users), 500)

test_users = [user for user in all_users if user not in train_users]
val_test_users = [user for user in all_users if user not in train_users]
test_users = random.sample(list(val_test_users), 400)
val_users = [user for user in val_test_users if user not in test_users]

# 创建一个字典，按用户编号分组
# test_users = random.sample(test_users, 60)

# 创建一个字典，按用户编号分组
user_to_signatures = {user: {'genuine': [], 'forged': []} for user in unique_users}

# 遍历数据并分组
for i in range(len(x)):
    user = y[i]
    forgery = yforg[i]
    if forgery == 0:
        user_to_signatures[user]['genuine'].append(x[i])
    else:
        user_to_signatures[user]['forged'].append(x[i])

# 生成训练和测试样本
def generate_pairs(user_to_signatures, selected_users, num_signatures=24):
    genuine_pairs = []
    all_fake_pairs = []
    fake_pairs = []

    # 1. 生成所有真签名配对（正样本对）
    for user in selected_users:
        genuine_signatures = user_to_signatures[user]['genuine']
        # genuine_signatures = random.sample(genuine_signatures, num_signatures)
        if len(genuine_signatures) >= 2:# 原本是24真
            # 使用 itertools.combinations 生成所有的真签名配对
            genuine_pairs.extend(itertools.combinations(genuine_signatures, 2))

    # 2. 生成所有真签名和伪造签名的配对（负样本对）
        forged_signatures = user_to_signatures[user]['forged']
        
        if len(genuine_signatures) > 0 and len(forged_signatures) > 0:  # 30假
            # 使用 itertools.product 生成所有真签名和伪造签名的配对
            fake_pairs_for_user = list(itertools.product(genuine_signatures, forged_signatures))
            all_fake_pairs.extend(fake_pairs_for_user)

    # 3. 随机抽样负样本对，保证数量与正样本对一致
    fake_pairs = random.sample(all_fake_pairs, len(genuine_pairs))

    return genuine_pairs, fake_pairs

class SignaturePairDataset(Dataset):
    def __init__(self, genuine_pairs, fake_pairs, transform=None):
        self.genuine_pairs = genuine_pairs
        self.fake_pairs = fake_pairs
        self.pairs = genuine_pairs + fake_pairs  # 合并正负样本对
        self.labels = [1] * len(genuine_pairs) + [0] * len(fake_pairs)  # 1表示正样本，0表示负样本
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        label = self.labels[idx]
        
        # 将 numpy 数组转换为 PIL 图像
        img1 = Image.fromarray(img1.squeeze())
        img2 = Image.fromarray(img2.squeeze())

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

# 定义数据增强或预处理方法（可选）
from torchvision import transforms

# 定义转换操作
# 数据增强
def binarize(img):
    return img.point(lambda p: 255 if p > 0 else 0)

train_transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomCrop(224),
    # ImageOps.invert,  # 先反色
    # binarize,         # 再二值化
    transforms.ToTensor()  # 将 PIL 图像转换为 torch.Tensor 并归一化到 [0, 1]
])



test_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),

        transforms.ToTensor()  # 将 PIL 图像转换为 torch.Tensor 并归一化到 [0, 1]
    ])


# train_transform = transforms.Compose([
#     # 1. 基础预处理
#     transforms.Grayscale(num_output_channels=1),  # 确保灰度
    
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
#     # transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度标准化
# ])

# test_transform = transforms.Compose([
#     # 1. 基础预处理（与训练集一致）
#     transforms.Resize([224, 224]),  # 直接缩放到目标尺寸（无需RandomCrop）
#     transforms.Grayscale(num_output_channels=1),
    
#     # 2. 仅包含确定性转换
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.5], std=[0.5])  # 必须与训练集相同
# ])

train_genuine_pairs, train_fake_pairs = generate_pairs(user_to_signatures, train_users)
val_genuine_pairs, val_fake_pairs = generate_pairs(user_to_signatures, val_users)
test_genuine_pairs, test_fake_pairs = generate_pairs(user_to_signatures, val_test_users)

# 创建数据集和DataLoader
train_dataset = SignaturePairDataset(train_genuine_pairs, train_fake_pairs, transform=train_transform)
val_dataset = SignaturePairDataset(val_genuine_pairs, val_fake_pairs, transform=test_transform)
test_dataset = SignaturePairDataset(test_genuine_pairs, test_fake_pairs, transform=test_transform)


train_loader = DataLoader(train_dataset, batch_size=runtime_batch_size(16), shuffle=True, num_workers=runtime_workers(8), pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=runtime_batch_size(16), shuffle=True, num_workers=runtime_workers(8), pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=runtime_batch_size(64), shuffle=False, num_workers=runtime_workers(8), pin_memory=True)
