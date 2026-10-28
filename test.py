import os
import sys
# 获取当前文件的绝对路径
file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
directory = os.path.dirname(file_path)
# 改变当前工作目录
os.chdir(directory)

from model import ZYCFocalLoss
from data import get_data_loader
from PIL import ImageOps, Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from metrics import accuracy,calculate_auc
from argparse import ArgumentParser
from runtime_paths import configure_runtime
from models.graphsignet import Mix_Model
from collections import OrderedDict
# from thop import profile


import numpy as np
# import UTSIG

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def save_gray_and_patched_images(
    tensor: torch.Tensor,
    original_dir: str = "original_gray_images",
    patched_dir: str = "patched_gray_images_with_gap",
    gap: int = 2,  # 块之间的空距（像素），可调整
    normalize: bool = True,  # 是否自动归一化到[0,255]（适配模型输出/原始图像）
    pad_len: int = 4  # 文件名序号补零位数（如0001、0002）
):
    """
    处理B,1,224,224的Tensor，保存两类灰度图：
    1. 原始灰度图 → original_dir
    2. 14×14切块+空距拼接图 → patched_dir

    参数说明：
    - tensor: 输入Tensor，形状必须为 (B, 1, 224, 224)
    - original_dir: 原始灰度图保存文件夹
    - patched_dir: 切块拼接图保存文件夹
    - gap: 块之间的空距（默认2像素，可改为1、3等）
    - normalize: 是否自动归一化到[0,255]（若Tensor已是[0,255]uint8，设为False）
    - pad_len: 文件名序号补零（如pad_len=4 → 0001.png、0002.png）
    """
    # 1. 验证Tensor形状
    assert tensor.shape[1] == 1 and tensor.shape[2] == 224 and tensor.shape[3] == 224, \
        f"Tensor形状必须是 (B,1,224,224)，当前是 {tensor.shape}"
    B = tensor.shape[0]  # 样本数

    # 2. 创建保存文件夹（不存在则创建）
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(patched_dir, exist_ok=True)

    # 3. 遍历每个样本处理
    for idx in range(B):
        # ---------------------- 步骤1：保存原始灰度图 ----------------------
        # 提取单张图（去掉通道维），并转移到CPU（兼容GPU张量）
        img = tensor[idx].squeeze(0).cpu()  # 形状：(224, 224)
        
        # 处理值范围：归一化到[0,255]并转为uint8（灰度图标准格式）
        if normalize:
            # 先归一化到[0,1]（避免原始值范围异常），再转[0,255]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 加小值防除零
            img_np = (img.numpy() * 255).astype(np.uint8)
        else:
            # 直接转为uint8（需确保Tensor值已在[0,255]）
            img_np = img.numpy().astype(np.uint8)
        
        # 保存原始图（按序号命名）
        original_path = os.path.join(original_dir, f"{idx:0{pad_len}d}.png")
        Image.fromarray(img_np).save(original_path)

        # ---------------------- 步骤2：生成14×14切块+空距拼接图 ----------------------
        patch_size = 224 // 14  # 每个块的尺寸：224/14=16 → (16,16)
        num_patches = 14  # 14×14个块

        # 计算拼接图的总尺寸：块尺寸×数量 + 空距×(块数-1)（14块有13个间隔）
        total_size = num_patches * patch_size + (num_patches - 1) * gap

        # 创建白色背景画布（空距部分填充白色，也可改为0设为黑色）
        patched_canvas = np.ones((total_size, total_size), dtype=np.uint8) * 255

        # 遍历每个块，放到画布对应位置
        for i in range(num_patches):  # 行方向（垂直方向，对应图像高度）
            for j in range(num_patches):  # 列方向（水平方向，对应图像宽度）
                # 1. 计算原始图中当前块的坐标（左上角→右下角）
                y_start = i * patch_size
                y_end = y_start + patch_size
                x_start = j * patch_size
                x_end = x_start + patch_size

                # 2. 提取当前块
                patch = img_np[y_start:y_end, x_start:x_end]

                # 3. 计算画布上当前块的坐标（考虑空距）
                canvas_y_start = i * (patch_size + gap)
                canvas_y_end = canvas_y_start + patch_size
                canvas_x_start = j * (patch_size + gap)
                canvas_x_end = canvas_x_start + patch_size

                # 4. 将块放入画布
                patched_canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = patch

        # 保存拼接图（与原始图同名，不同文件夹）
        patched_path = os.path.join(patched_dir, f"{idx:0{pad_len}d}.png")
        Image.fromarray(patched_canvas).save(patched_path)

    print(f"处理完成！")
    print(f"原始灰度图：{os.path.abspath(original_dir)}（共{B}张）")
    print(f"切块拼接图：{os.path.abspath(patched_dir)}（共{B}张）")
    print(f"块大小：{patch_size}×{patch_size}，块间空距：{gap}像素")


@torch.no_grad()
def eval(model, criterion, dataloader, log_interval=50):
    model.eval()
    running_Loss = 0
    running_loss_g = 0
    running_loss_tokens = 0

    loss_tokens = 0
    
    number_samples = 0

    distances = []

    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        b = x1.shape[0]
        x = torch.cat((x1,x2), dim=0)
        # x, tokens, total_codebook_loss = model(x)

        # save_gray_and_patched_images(x)
        x = model(x)
        x1, x2 = x[:b], x[b:]
        # tokens1, tokens2 = tokens[:b], tokens[b:]

        # x1, x2 = torch.cat((x1,tokens1), dim=1), torch.cat((x2,tokens2), dim=1)
        loss_g = criterion(x1, x2, y)
        # loss_tokens = criterion(tokens1, tokens2, y)

        Loss = loss_g# + 0.2*loss_tokens + 0.1*total_codebook_loss

        number_samples += len(x1)
        running_Loss += Loss.item() * len(x1)
        running_loss_g += loss_g.item() * len(x1)
        running_loss_tokens += loss_tokens * len(x1)

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}, loss_g: {:.4f}, loss_tokens: {:.4f}'.format(\
                batch_idx+1, len(dataloader), running_Loss / number_samples, running_loss_g / number_samples,\
                      running_loss_tokens / number_samples))
        distances.extend(zip((10*(1-F.cosine_similarity(x1, x2, dim=1)+1e-6)).cpu().tolist(), y.cpu().tolist()))

    distances, y = zip(*distances)
    distances, y = torch.tensor(distances), torch.tensor(y)
    max_accuracy, threshold = accuracy(distances, y)
    FRR,FAR,AUC,EER = calculate_auc(distances, y, threshold)
    print(f'Max accuracy: {max_accuracy}')
    print(f'threshold:{threshold}')
    print(f'FRR:{FRR:.2%}  FAR:{FAR:.2%}  EER:{EER:.2%}  AUC:{AUC:.2%}\n')

    return max_accuracy, EER

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--batch_size', '--batch-size', type=int, default=16)
    parser.add_argument('--data-root', default='data', help='Root directory of the prepared datasets')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default=None, help='For example: cpu, cuda:0')
    parser.add_argument('--dataset', type=str, nargs='+', choices=['cedar', 'bengali', 'hindi', 'gpds', 'chisig_all'], default=['bengali', 'hindi', 'chisig_all', 'gpds'])
    parser.add_argument('--checkpoint', required=True, help='Path to a trusted model checkpoint')
    parser.add_argument('--model', help='Model architecture', choices=['graphsignet'], default='graphsignet')
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=0.0015, metavar='LR',
                        help='learning rate (default: 0.002)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports(default: 20)')
    parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('batch size must be positive and workers must be nonnegative')
    configure_runtime(args)
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))


    print(args)
    # with open(filename + '.txt', "a") as file:
    #     file.write(f'{args}\n')

    model = Mix_Model().to(device)
    # 定义输入
    # input = torch.randn(1, 1, 224, 224).to(device)

    # 计算FLOPS和参数数量
    # flops, params = profile(model, inputs=(input,))

    # print(f"FLOPS: {flops}")   #FLOPS: 4,731,490,304.0
    # print(f"Params: {params}")  #Params: 7,771,296.0
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"参数量: {count_parameters(model):,}")# 参数量: 8,072,352; [2,2,2]EMD: 8,269,728, 哈夫曼：15,792,160 


    choices = list(args.dataset)
    path = os.path.expanduser(args.checkpoint)
    if not os.path.isfile(path):
        parser.error(f'Checkpoint not found: {path}')
    print('Loading pre-trained model')
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict['model'])

    for i in choices:
        args.dataset = [i]
        print(f"数据集是{i}")

        test_transform = transforms.Compose([
            # transforms.Resize([224, 224]),
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            ImageOps.invert,
            transforms.ToTensor(),
        ])
        if args.dataset == ['cedar']:
            import CEDAR
            testloader = CEDAR.test_loader
        elif args.dataset == ['chisig_all']:
            import ChiSig_all
            trainloader = ChiSig_all.train_loader
            testloader = ChiSig_all.val_loader
            
        elif args.dataset == ['gpds']:
            import GPDS
            testloader = GPDS.test_loader
            # testloader = GPDS.test_loader
        else:
            testloader = get_data_loader(is_train=False, is_val=False, batch_size=args.batch_size, image_transform=test_transform, datasets=args.dataset)

        

        if args.dataset == ['cedar'] or args.dataset == ['gpds']:
            a, ma, b, mb = 0.3 ,0.3, 1.0, 1.0
        elif args.dataset == ['hindi']:
            a, ma, b, mb = 0.5 ,0.5, 1.8, 1.8
        elif args.dataset == ['chisig_all']:
            a, ma, b, mb = 0.3 ,0.3, 1.2, 1.2
        else:#bengali
            a, ma, b, mb = 0.5 ,0.5, 1.8, 1.8

        print(f'a={a}, ma={ma}, b={b}, mb={mb}')

        # print('0.3*(x-0.3)^1.5, (1/x-1/1)*0.1')
        # 0.3*(x-1)^1.5, (1/x-1/5)*2
        criterion = ZYCFocalLoss(a=a ,ma=ma, b=b, mb=mb).to(device)
        # print(model)

        print('Testing', '-'*20)
        # with open(filename + '.txt', "a") as file:
        #     file.write('Evaluating'+'-'*20+'\n')

        model.eval()
        acc,eer =  eval(model, criterion, testloader)

    # print('Saving checkpoint..')

    # logdir = f'checkpoints/graph_signet/{filename}/max_acc_checkpoint.pt'
    # torch.save(max_acc_save, logdir)
