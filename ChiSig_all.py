from runtime_paths import data_path, runtime_batch_size, runtime_workers
import os
import re
import random
import itertools
from PIL import ImageOps, Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict

# 设置路径
root_dir = data_path('ChiSig_all_C')
pattern = re.compile(r'^(.+)-(\d+)-(\d+)\.png$') # 姓名-id-序号.jpg

# 数据结构：user_key -> {'genuine': [path1, ...], 'forged': [path2, ...]}
user_to_signatures = defaultdict(lambda: {'genuine': [], 'forged': []})

# --- 1. 收集数据 (保持不变) ---
for fname in os.listdir(root_dir):
    match = pattern.match(fname)
    if not match:
        continue
    name, id_str, idx = match.groups()
    id_num = int(id_str)
    key = f"{name}-{id_num}" 

    if id_num >= 100:
        # 假签名：id=xxx+100 => 真用户为 xxx
        real_key = f"{name}-{id_num - 100}"
        # 假签名关联到它所伪造的真实用户
        user_to_signatures[real_key]['forged'].append(os.path.join(root_dir, fname))
    else:
        # 真签名
        user_to_signatures[key]['genuine'].append(os.path.join(root_dir, fname))
# -----------------------------


# 🚀 2. 过滤出所有真签名数量 >= 2 的用户
all_users_for_training = {
    k: v for k, v in user_to_signatures.items()
    if len(v['genuine']) >= 2
}

# 3. 划分用户 Key (Train/Val Split)
all_user_keys = list(all_users_for_training.keys())
random.seed(1)
random.shuffle(all_user_keys)

num_total = len(all_user_keys)
num_train = int(num_total * 0.5)
num_val = num_total - num_train

train_users = all_user_keys[:num_train]
val_users = all_user_keys[num_train:]


# 🔑 4. **核心修复：创建独立的数据字典**
#    确保训练集和验证集数据**严格隔离**，杜绝数据泄露。
train_data = {k: all_users_for_training[k] for k in train_users}
val_data = {k: all_users_for_training[k] for k in val_users}

print(f"参与训练/验证的总用户数: {num_total}")
print(f"训练集用户数: {len(train_users)}")
print(f"验证集用户数: {len(val_users)}")


# 🔄 5. 配对函数：依赖传入的独立字典
def generate_pairs_adaptive(user_data_dict):
    """
    根据传入的 user_data_dict 生成签名对。
    Args:
        user_data_dict (dict): 独立的数据字典（train_data 或 val_data）。
    """
    genuine_pairs = []
    fake_pairs = []

    for user in user_data_dict:
        data = user_data_dict[user]
        genuine = data['genuine']
        forged = data['forged']

        # 1. 真签名对 (Genuine Pair)
        genuine_pairs.extend(itertools.combinations(genuine, 2))

        # 2. 伪造对 (Fake Pair)
        if forged:
            fake_pairs.extend(itertools.product(genuine, forged))

    # 可选：平衡正负样本数
    if len(fake_pairs) > len(genuine_pairs):
        # 如果假签名对多，则从假签名对中随机采样，使其数量与真签名对持平
        fake_pairs = random.sample(fake_pairs, len(genuine_pairs))
    return genuine_pairs, fake_pairs


# 6. 定义 Dataset (保持不变)
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

        return img1, img2, label

def binarize(img):
    return img.point(lambda p: 255 if p > 0 else 0)
# 7. Transforms 和 DataLoader (保持不变)
train_transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomCrop(224),
    ImageOps.invert, 
    # binarize, 
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    ImageOps.invert,
    # binarize,
    transforms.ToTensor()
])

# 构建 Dataset 和 DataLoader：传入独立字典
train_g, train_f = generate_pairs_adaptive(train_data)
val_g, val_f = generate_pairs_adaptive(val_data)

train_dataset = SignaturePairDataset(train_g, train_f, transform=train_transform)
val_dataset = SignaturePairDataset(val_g, val_f, transform=test_transform)


train_loader = DataLoader(train_dataset, batch_size=runtime_batch_size(16), shuffle=True, num_workers=runtime_workers(8), pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=runtime_batch_size(16), shuffle=False, num_workers=runtime_workers(8), pin_memory=True)


# 打印数据集大小
print("--- 数据集大小 (Pair Count) ---")
print(f"训练集总样本数: {len(train_dataset)} (真: {len(train_g)}, 假: {len(train_f)})")
print(f"验证集总样本数: {len(val_dataset)} (真: {len(val_g)}, 假: {len(val_f)})")
