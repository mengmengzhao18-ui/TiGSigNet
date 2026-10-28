from runtime_paths import data_path, runtime_workers
import os
import numpy as np
import pandas as pd
import itertools
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Dict, Union

seed = 42
np.random.seed(seed)

def get_data_loader(is_train, is_val, batch_size, image_transform, datasets=['cedar']):
    # 初始化数据集列表
    datasets_list = []
    
    for dataset in datasets:
        if dataset == 'cedar':
            data_dir = data_path('CEDAR_C')
        elif dataset == 'bengali':
            data_dir = data_path('BHSig260', 'Bengali_C')
        elif dataset == 'hindi':
            data_dir = data_path('BHSig260', 'Hindi_C')
        else:
            raise ValueError(f'Unknown dataset {dataset}')

        # 检查数据集分割文件
        if not os.path.exists(os.path.join(data_dir, 'train.csv')) or not os.path.exists(os.path.join(data_dir, 'test.csv')):
            print(f'Warning: Train/test splits not found for {dataset}, skipping this dataset')
            

        # 创建当前数据集实例
        current_dataset = SignDataset(is_train, is_val, data_dir, image_transform)
        datasets_list.append(current_dataset)
    
    # 如果没有有效数据集，抛出异常
    if not datasets_list:
        raise ValueError("No valid datasets provided")
    
    # 合并所有数据集
    combined_dataset = ConcatDataset(datasets_list)
    
    # 创建数据加载器
    is_shuffle = is_train 
    loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=is_shuffle, 
                        num_workers=runtime_workers(8), pin_memory=True)
    return loader

class SignDataset(Dataset):
    def __init__(self, is_train: bool, is_val: bool, data_dir: str, image_transform=None):
        if not os.path.exists(os.path.join(data_dir, 'train1.csv')) or not os.path.exists(os.path.join(data_dir, 'test1.csv')):
            print('Not found train/test splits, run create_annotation first')
        else:
            print('Use existed train/test splits')
        
        if is_train:
            self.df = pd.read_csv(os.path.join(data_dir, 'trainDS.csv'), header=None)
        elif is_val:
            self.df = pd.read_csv(os.path.join(data_dir, 'valDS.csv'), header=None)
        else:
            self.df = pd.read_csv(os.path.join(data_dir, 'test_allDS.csv'), header=None)

        self.image_transform = image_transform
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x1, x2, y = self.df.iloc[index]

        x1 = Image.open(x1).convert('L')
        x2 = Image.open(x2).convert('L')

        if self.image_transform:
            x1 = self.image_transform(x1)
            x2 = self.image_transform(x2)
        
        return x1, x2, y

def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray]:
    """ Loads a dataset that was pre-processed in a numpy format

    Parameters
    ----------
    path : str
        The path to a .npz file, containing attributes "x", "y", "yforg",
        "usermapping" and "filenames"

    Returns
    -------
    x : np.ndarray (N x 1 x H x W) are N grayscale signature images of size H x W
    y : np.ndarray (N) indicating the user that wrote the signature
    yforg : np.ndarray (N) indicating wether the signature is a forgery
    user_mapping: dict, mapping the indexes in "y" with the original user
                 numbers from the dataset
    filenames: np.ndarray(str) (N), the filename associated with the signature image
    -------

    """
    with np.load(path, allow_pickle=True) as data:
        x, y, yforg = data['x'], data['y'], data['yforg']
        user_mapping, filenames = data['user_mapping'], data['filenames']

    return x, y, yforg, user_mapping, filenames
