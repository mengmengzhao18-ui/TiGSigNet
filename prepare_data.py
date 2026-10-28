import glob
import sys
import os
file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
directory = os.path.dirname(file_path)
# 改变当前工作目录
os.chdir(directory)
import random
import itertools
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import csv


def make_partition(
    signers: List[int],
    pair_genuine_genuine: List[Tuple[int, int]],
    pair_genuine_forged: List[Tuple[int, int]],

):
    samples = []
    for signer_id in signers:
        sub_pair_genuine_forged = random.sample(pair_genuine_forged, len(pair_genuine_genuine))

        genuine_genuine = list(itertools.zip_longest(pair_genuine_genuine, [], fillvalue=1)) # y = 1
        genuine_genuine = list(map(lambda sample: (signer_id, *sample[0], sample[1]), genuine_genuine))
        samples.extend(genuine_genuine)
        genuine_forged = list(itertools.zip_longest(sub_pair_genuine_forged, [], fillvalue=0)) # y = 0
        genuine_forged = list(map(lambda sample: (signer_id, *sample[0], sample[1]), genuine_forged))
        samples.extend(genuine_forged)
    return samples

def write_csv(file_path, samples):
    with open(file_path, 'wt') as f:
        writer = csv.writer(f)
        writer.writerows(samples)

def prepare_CEDAR(M: int, K: int, random_state=0, data_dir='data/CEDAR'):
    def get_path(row):
        writer_id, x1, x2, y = row
        if y == 1:
            x1 = os.path.join(data_dir, 'full_org', f'original_{writer_id}_{x1}.png')
            x2 = os.path.join(data_dir, 'full_org', f'original_{writer_id}_{x2}.png')
        else:
            x1 = os.path.join(data_dir, 'full_org', f'original_{writer_id}_{x1}.png')
            x2 = os.path.join(data_dir, 'full_forg', f'forgeries_{writer_id}_{x2}.png')
        return x1, x2, y # drop writer_id

    random.seed(random_state)
    signers = list(range(1, K+1))
    num_genuine_sign = 24
    num_forged_sign = 24

    train_signers, test_val_signers = train_test_split(signers, test_size=K-M, random_state=random_state)

    val_signers = random.sample(test_val_signers, 5)
    val_signers = [int(signer) for signer in val_signers]
    test_signers = [int(signer) for signer in test_val_signers if signer not in val_signers]

    pair_genuine_genuine = list(itertools.combinations(range(1, num_genuine_sign+1), 2))
    pair_genuine_forged = list(itertools.product(range(1, num_genuine_sign+1), range(1, num_forged_sign+1)))

    train_samples = make_partition(train_signers, pair_genuine_genuine, pair_genuine_forged)
    train_samples = list(map(get_path, train_samples))
    write_csv(os.path.join(data_dir, f'train{str(random_state)}.csv'), train_samples)
    val_samples = make_partition(val_signers, pair_genuine_genuine, pair_genuine_forged)
    val_samples = list(map(get_path, val_samples))
    write_csv(os.path.join(data_dir, f'val{str(random_state)}.csv'), val_samples)
    test_samples = make_partition(test_signers, pair_genuine_genuine, pair_genuine_forged)
    test_samples = list(map(get_path, test_samples))
    write_csv(os.path.join(data_dir, f'test{str(random_state)}.csv'), test_samples)


def prepare_BHSig260(M: int, K: int, random_state=0, data_dir='data/BHSig260/Bengali',dataset='B'):
    def get_path(row):
        writer_id, x1, x2, y = row
        if y == 1:
            x1 = os.path.join(data_dir, f'{writer_id:03d}', f'{dataset}-S-{writer_id}-G-{x1:02d}.tif')
            x2 = os.path.join(data_dir, f'{writer_id:03d}', f'{dataset}-S-{writer_id}-G-{x2:02d}.tif')
        else:
            x1 = os.path.join(data_dir, f'{writer_id:03d}', f'{dataset}-S-{writer_id}-G-{x1:02d}.tif')
            x2 = os.path.join(data_dir, f'{writer_id:03d}', f'{dataset}-S-{writer_id}-F-{x2:02d}.tif')
        return x1, x2, y # drop writer_id

    random.seed(random_state)
    signers = list(range(1, K+1))
    num_genuine_sign = 24
    num_forged_sign = 30

    train_signers, test_val_signers = train_test_split(signers, test_size=K-M, random_state=random_state)
    test_val_signers = [138.0, 116.0, 103.0, 97.0, 11.0, 121.0, 37.0, 148.0, 79.0, 56.0, 45.0, 150.0, 18.0, 63.0, 100.0, 83.0, 107.0, 112.0, 101.0, 77.0, 51.0, 62.0, 92.0, 14.0, 65.0, 111.0, 87.0, 67.0, 19.0, 27.0, 41.0, 84.0, 22.0, 1.0, 4.0, 49.0, 118.0, 140.0, 5.0, 149.0, 132.0, 55.0, 80.0, 26.0, 72.0, 40.0, 115.0, 99.0, 94.0, 117.0, 110.0, 24.0, 13.0, 2.0, 144.0, 106.0, 29.0, 7.0, 81.0, 124.0]
    # test_val_signers = [45.0, 2.0, 7.0, 22.0, 9.0, 1.0, 46.0, 31.0, 15.0, 42.0, 40.0, 19.0, 16.0, 36.0, 83.0, 25.0, 26.0, 39.0, 21.0, 44.0, 89.0, 30.0, 96.0, 90.0, 17.0, 74.0, 8.0, 18.0, 43.0, 24.0, 41.0, 20.0, 66.0, 76.0, 10.0, 75.0, 32.0, 23.0, 47.0, 48.0, 28.0, 49.0, 33.0, 6.0, 51.0, 53.0, 86.0, 37.0, 57.0, 65.0]
    train_signers = [int(signer) for signer in signers if signer not in test_val_signers]

    val_signers = random.sample(test_val_signers, 10)
    val_signers = [int(signer) for signer in val_signers]
    test_signers = [int(signer) for signer in test_val_signers if signer not in val_signers]

    # val_signers = [int(signer) for signer in test_val_signers]

    pair_genuine_genuine = list(itertools.combinations(range(1, num_genuine_sign+1), 2))
    pair_genuine_forged = list(itertools.product(range(1, num_genuine_sign+1), range(1, num_forged_sign+1)))

    train_samples = make_partition(train_signers, pair_genuine_genuine, pair_genuine_forged)
    train_samples = list(map(get_path, train_samples))
    write_csv(os.path.join(data_dir, f'trainDS.csv'), train_samples)
    val_samples = make_partition(val_signers, pair_genuine_genuine, pair_genuine_forged)
    val_samples = list(map(get_path, val_samples))
    write_csv(os.path.join(data_dir, f'valDS.csv'), val_samples)
    test_samples = make_partition(test_signers, pair_genuine_genuine, pair_genuine_forged)
    test_samples = list(map(get_path, test_samples))
    write_csv(os.path.join(data_dir, f'testDS.csv'), test_samples)


if __name__ == "__main__":
    # print('Preparing CEDAR dataset..')
    # prepare_CEDAR(M=45, K=55)
    # print('Preparing Bengali dataset..')
    # prepare_BHSig260(M=80, K=100, random_state=1, data_dir='data/BHSig260/Bengali_C',dataset='B')
    # print('Preparing Hindi dataset..')
    prepare_BHSig260(M=100, K=160, random_state=99, data_dir='data/BHSig260/Hindi_C',dataset='H')
    print('Done')
