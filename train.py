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

import numpy as np
import random

seed = 99
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(model, optimizer, criterion,  dataloader, log_interval=50):
    model.train()
    running_Loss = 0
    running_loss_g = 0
    running_loss_tokens = 0
    loss_tokens = 0

    number_samples = 0

    lr = optimizer.param_groups[0]['lr']
    print(f'lr = {lr}')
    # with open(filename + '.txt', "a") as file:
    #     file.write(f'lr = {lr}\n')

    distances = []
    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # optimizer_t.zero_grad()
        optimizer.zero_grad()

        b = x1.shape[0]
        x = torch.cat((x1,x2), dim=0)
        
        # x, tokens, total_codebook_loss = model(x)
        x = model(x)
        x1, x2 = x[:b], x[b:]
        # tokens1, tokens2 = tokens[:b], tokens[b:]

        # x1, x2 = torch.cat((x1,tokens1), dim=1), torch.cat((x2,tokens2), dim=1)
        loss_g = criterion(x1, x2, y)
        # loss_tokens = criterion(tokens1, tokens2, y)

        Loss = loss_g# + 0.2*loss_tokens + 0.1*total_codebook_loss

        Loss.backward()

        # optimizer_t.step()
        optimizer.step()


        number_samples += len(x1)
        running_Loss += Loss.item() * len(x1)
        running_loss_g += loss_g.item() * len(x1)
        running_loss_tokens += loss_tokens * len(x1)

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}, loss_g: {:.4f}, loss_tokens: {:.4f}'.format(\
                batch_idx+1, len(dataloader), running_Loss / number_samples, running_loss_g / number_samples,\
                      running_loss_tokens / number_samples))

            running_Loss = 0
            running_loss_g = 0
            running_loss_tokens = 0
            number_samples = 0

        distances.extend(zip((10*(1-F.cosine_similarity(x1, x2, dim=1)+1e-6)).cpu().tolist(), y.cpu().tolist()))

    distances, y = zip(*distances)
    distances, y = torch.tensor(distances), torch.tensor(y)
    max_accuracy, threshold = accuracy(distances, y)
    FRR,FAR,AUC,EER = calculate_auc(distances, y, threshold)
    print(f'Max accuracy: {max_accuracy}')
    print(f'threshold:{threshold}')
    print(f'FRR:{FRR:.2%}  FAR:{FAR:.2%}  EER:{EER:.2%}  AUC:{AUC:.2%}\n')

    return AUC

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
    filename = 'tigsignet'

    parser = ArgumentParser()
    parser.add_argument('--batch_size', '--batch-size', type=int, default=16)
    parser.add_argument('--data-root', default='data', help='Root directory of the prepared datasets')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default=None, help='For example: cpu, cuda:0')
    parser.add_argument('--dataset', type=str, nargs=1, choices=['cedar', 'chisig_all', 'bengali', 'hindi', 'gpds'], default=['chisig_all'])
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--run-name', default='tigsignet')
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
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
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
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
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
    filename = args.run_name
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))


    print(args)
    # with open(filename + '.txt', "a") as file:
    #     file.write(f'{args}\n')

    # from htcsignet import SigTransformer
    # model = SigTransformer(path).to(device)
    model = Mix_Model().to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"参数量: {count_parameters(model):,}")# 参数量: 8,072,352; [2,2,2]EMD: 8,269,728, 哈夫曼：15,792,160 

    # 继续训练用的
    # if os.path.exists(path):
    #     print('Loading pre-trained model')
    #     state_dict = torch.load(path, map_location=lambda storage, loc: storage)

    #     model.load_state_dict(state_dict['model'])



    if args.dataset == ['cedar'] or args.dataset == ['gpds']:
        a, ma, b, mb = 0.3 ,0.3, 1.0, 1.0
    elif args.dataset == ['hindi']:
        a, ma, b, mb = 0.5 ,0.5, 1.8, 1.8
    elif args.dataset == ['chisig_all']:
        a, ma, b, mb = 0.3 ,0.3, 1.2, 1.2
    else:#bengali
        a, ma, b, mb = 0.5 ,0.5, 1.8, 1.8


    # chi数据集需要额外降低学习率
    
    print(f'a={a}, ma={ma}, b={b}, mb={mb}')
    # print('0.3*(x-0.3)^1.5, (1/x-1/1)*0.1')
    criterion = ZYCFocalLoss(a=a ,ma=ma, b=b, mb=mb).to(device)

    optimizer = create_optimizer(args, model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    train_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop(224),
        ImageOps.invert,
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        ImageOps.invert,
        transforms.ToTensor(),
    ])


    if args.dataset == ['cedar']:
        import CEDAR
        trainloader = CEDAR.train_loader
        valloader = CEDAR.test_loader
    elif args.dataset == ['gpds']:
        import GPDS
        print("数据集是GPDS")
        trainloader = GPDS.train_loader
        valloader = GPDS.val_loader
        # testloader = GPDS.test_loader
    elif args.dataset == ['chisig_all']:
        import ChiSig_all
        print("数据集是ChiSig_all")
        trainloader = ChiSig_all.train_loader
        valloader = ChiSig_all.val_loader
    else:
        trainloader = get_data_loader(is_train=True, is_val=False, batch_size=args.batch_size, image_transform=train_transform, datasets=args.dataset)
        valloader = get_data_loader(is_train=False, is_val=True, batch_size=args.batch_size, image_transform=test_transform, datasets=args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    model.train()
    print(model)

    min_eer = 1
    directory = os.path.join(args.output_dir, 'graph_signet', filename)
    if not os.path.exists(directory):
        os.makedirs(directory) 
    for epoch in range(num_epochs):

        # if epoch < 19:  # 继续训练用的
        #     lr_scheduler.step(epoch=epoch)
        #     continue

        AUC = 1
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('Training', '-'*20)
        # with open(filename + '.txt', "a") as file:
        #     file.write('Epoch {}/{}\n'.format(epoch, num_epochs))
        #     file.write('Training'+'-'*20+'\n')

        AUC = train(model, optimizer, criterion, trainloader)
        print('Evaluating', '-'*20)
        # with open(filename + '.txt', "a") as file:
        #     file.write('Evaluating'+'-'*20+'\n')

        if AUC<0.99:
            print("low auc, no eval\n")
            # with open(filename + '.txt', "a") as file:
            #     file.write("low auc, no eval"+'\n')
        else:
            acc,eer =  eval(model, criterion, valloader)
            # acc,eer =  eval(model, criterion, testloader)
            # if acc > max_accuracy:
            #     max_accuracy = acc
            #     max_acc_save = {
            #         'model': model.state_dict(),
            #         'scheduler': lr_scheduler.state_dict(),
            #         'optim': optimizer.state_dict(),
            #     }
            save = {
                'model': model.state_dict(),
            }
            logdir = os.path.join(directory, f'epoch{epoch+1}_acc{acc:.4%}_eer{eer:.4%}.pt')
            if epoch >= num_epochs-6:
                torch.save(save, logdir)
        lr_scheduler.step(epoch=epoch)
        # lr_scheduler_t.step()
        optimizer.zero_grad()

        # print('Saving checkpoint..')

        # logdir = f'checkpoints/graph_signet/{filename}/max_acc_checkpoint.pt'
        # torch.save(max_acc_save, logdir)
