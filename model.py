import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torchvision import models

class ContrastiveLoss(nn.Module):
    def __init__(self, alpha, beta, margin):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, x1, x2, y):
        '''
        Shapes:
        -------
        x1: [B,C]
        x2: [B,C]
        y: [B,1]

        Returns:
        --------
        loss: [B,1]]
        '''
        distance = torch.pairwise_distance(x1, x2, p=2)
        loss = self.alpha * (1-y) * distance**2 + \
               self.beta * y * (torch.max(torch.zeros_like(distance), self.margin - distance)**2)
        return torch.mean(loss, dtype=torch.float)

class MarginBasedFocalLoss(nn.Module):
    def __init__(self,s,a,ma,b,mb):
        super().__init__()
        self.s = s
        self.a = a
        self.ma = ma
        self.b = b
        self.mb = mb

    def forward(self,x1,x2,y):
        # 1是真，0是假
        distance = torch.pairwise_distance(x1, x2, p=2)
        loss = nn.functional.sigmoid(self.s * (distance - self.a)) * y * (torch.max(torch.zeros_like(distance), distance - self.ma) ** 2) + \
                 nn.functional.sigmoid(self.s * (self.b - distance)) * (1-y) * (torch.max(torch.zeros_like(distance), self.mb - distance) ** 2)
        return torch.mean(loss, dtype=torch.float)
    
class ZYCFocalLoss(nn.Module):
    def __init__(self, a=1 ,ma=0.7, b=6, mb=7):
        super().__init__()
        self.a = a
        self.ma = ma
        self.b = b
        self.mb = mb

        # self.xa=0.3
        # self.xb=0.7
        # self.xma=0.25
        # self.xmb=0.75
        # self.xc=1.1
        # self.xfix = pow((self.xa+self.xb)/2,1.5)
    def forward(self,x1,x2,y):
        # 1是真，0是假

        # distance = torch.pairwise_distance(x1, x2, p=2)

        # loss_g = 0.0

        # d_true = torch.where(distance > self.xa,
        #                     distance - self.xma,
        #                     torch.zeros_like(distance))
        # # Compute loss_t for all distances
        # loss_t = d_true + torch.pow(d_true, 2)

        # # Compute loss_f for all distances
        # loss_f = torch.where(distance < self.xb,
        #                     ((1 / (distance)) - (1 / self.xmb)) * self.xfix,
        #                     torch.zeros_like(distance))
        # loss_f = torch.where(distance > self.xc,
        #                     F.sigmoid(distance - self.xc),
        #                     loss_f)
        
        # loss_g =  y * loss_t + \
        #     (1-y) * loss_f
        
        distance = F.cosine_similarity(x1, x2, dim=1)-1e-6
        distance = (1-distance) * 10
        loss = 0.0

        d_true = torch.where(distance > self.a,
                            distance - self.ma,
                            torch.zeros_like(distance))
        # Compute loss_t for all distances
        loss_t = 0.3*torch.pow(d_true, 1.5)

        # Compute loss_f for all distances
        # distance = torch.clamp(distance, min=1e-3)
        loss_f = torch.where(distance < self.b,
                            torch.max(torch.zeros_like(distance), ((1 / distance) - (1 / self.mb)) * 0.1),
                            torch.zeros_like(distance))
        # loss_f = torch.where(distance > self.c,
        #                     F.sigmoid(distance - self.c),
        #                     loss_f)
        
        loss =  y * loss_t + \
            (1-y) * loss_f

        return torch.mean(loss, dtype=torch.float)
        return torch.mean(loss + loss_g*0.5, dtype=torch.float)


class SigNet(nn.Module):
    '''
    Reference Keras: https://github.com/sounakdey/SigNet/blob/master/SigNet_v1.py
    '''
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            #input size = [155, 220, 1]
            nn.Conv2d(1, 96, 11), # size = [145,210]
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2), # size = [72, 105]
            nn.Conv2d(96, 256, 5, padding=2, padding_mode='zeros'), # size = [72, 105]
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2), # size = [36, 52]
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 384, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(2, stride=2), # size = [18, 26]
            nn.Dropout2d(p=0.3),
            nn.Flatten(1, -1), # 18*26*256
            nn.Linear(18*26*256, 1024),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 128),
        )

        # TODO: init bias = 0

    def forward(self, x1):
        x1 = self.features(x1)

        return x1
