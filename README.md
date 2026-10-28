<h1 align="center">TiGSigNet</h1>

<p align="center">
  <strong>Token-interaction Graph Signature Network<br>for Offline Signature Verification</strong>
</p>

<p align="center">离线手写签名验证 · 局部语义交互 · 全局图结构建模</p>

<p align="center">
  <a href="#模型概览">模型概览</a> ·
  <a href="#论文结果">论文结果</a> ·
  <a href="#可视化">可视化</a> ·
  <a href="#安装">安装</a> ·
  <a href="#训练">训练</a> ·
  <a href="#测试">测试</a>
</p>

## 模型概览

TiGSigNet 面向书写者无关的离线签名验证，通过两个并行分支联合学习签名的局部笔画细节与全局结构关系。

<p align="center">
  <img src="assets/architecture.png" width="1000" alt="论文图 1：TiGSigNet 整体架构，包含 SiT 分支、ViG 分支及跨分支构图引导">
</p>

*整体架构与构图过程，摘自论文 Fig. 1。点击图片可查看大图。*

- **SiT 分支**：通过 Token Aggregation and Broadcasting（TAB）聚合局部信息，并将语义信息广播回空间特征。
- **ViG 分支**：利用动态图卷积建模笔画之间的拓扑关系，由 SiT 特征引导 KNN 图的构建。
- **特征融合**：拼接 96 维 token 表征和 384 维图表征，形成 480 维签名表征，用于签名对比对。

## 论文结果

以下数值摘自论文 Tables IV–VII，单位为 **%**；它们是论文报告结果，尚未通过本仓库的完整训练复现实验核验。

| 数据集 | Accuracy ↑ | FRR ↓ | FAR ↓ | EER ↓ |
|:---|---:|---:|---:|---:|
| BHSig260-Bengali | 99.13 | 0.93 | 0.80 | 0.91 |
| BHSig260-Hindi | 98.21 | 1.54 | 2.04 | 1.82 |
| CEDAR | 99.96 | 0.00 | 0.07 | 0.04 |
| GPDS-Synthetic | 97.82 | 1.92 | 2.43 | 2.18 |
| ChiSig | 98.27 | 1.25 | 2.54 | 1.85 |

↑ 越高越好；↓ 越低越好。EER 为等错误率，FAR 为错误接受率，FRR 为错误拒绝率。

## 可视化

<p align="center">
  <img src="assets/graph-construction.png" width="760" alt="论文图 6：Bengali、CEDAR 和 ChiSig 的原始签名与动态图连接可视化">
</p>

*签名图构建可视化，摘自论文 Fig. 6：上排为签名图像，下排为对应的图连接。青色表示签名相关连接，深蓝色表示背景相关连接。图片来自论文，非本次运行生成。*

## 安装

先安装与你的设备匹配的 PyTorch 和 torchvision，再安装其他依赖：

```bash
python -m pip install -r requirements.txt
```

将准备好的数据放到 `data/` 目录，或通过 `--data-root` 指定数据根目录。数据集和训练权重不包含在本仓库中。

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
