import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import os
import numpy as np
from PIL import Image
import random
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(500),
            nn.Linear(500, 100),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(100, 6)
        )


    def forward(self, x):
        x = self.fc(x)
        # x = F.softmax(x, -1)
        return x

class Mynet22(nn.Module):
    def __init__(self):
        super(Mynet22, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1664, 2000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2000),
            nn.Linear(2000, 1000),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1000, CLASS_NUM)
        )


    def forward(self, x):
        x = self.fc(x)
        return x

class Mynet2(nn.Module):
    def __init__(self):
        super(Mynet2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2000),
            nn.Linear(2000, 1000),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 6)
        )


    def forward(self, x):
        x = self.fc(x)
        return x