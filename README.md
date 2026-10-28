# TiGSigNet

Source code for TiGSigNet: Token-interaction Graph Signature Network for Offline Signature Verification

## 安装

先安装与你的设备匹配的 PyTorch 和 torchvision，再安装其他依赖：

```bash
python -m pip install -r requirements.txt
```

将准备好的数据放到 `data/` 目录。数据集不包含在本仓库中。

## 训练

```bash
python train.py --dataset chisig_all --data-root data --device cuda:0 --batch-size 16 --workers 8 --epochs 25 --output-dir checkpoints --run-name chisig_all
```

支持的数据集：`cedar`、`bengali`、`hindi`、`gpds`、`chisig_all`。可运行 `python train.py --help` 查看参数。CPU 运行时设置 `--device cpu --workers 0`。

## 测试

```bash
python test.py --checkpoint "PATH_TO_CHECKPOINT.pt" --dataset cedar --data-root data --device cuda:0 --workers 8
```

可运行 `python test.py --help` 查看参数。同一权重可通过 `--dataset` 指定多个测试集。
