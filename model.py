import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch

import torch.optim as optim
import torch.nn.functional as F

# from torchinfo import summary


class Encoder(nn.Module):
  def __init__(self):
    super(Encoder,self).__init__()

    self.secret_conv1 = nn.Conv2d(3,8, kernel_size=3, padding="same")
    self.secret_conv2 = nn.Conv2d(8,16, kernel_size=3, padding="same")
    self.secret_conv3 = nn.Conv2d(16,32, kernel_size=3, padding="same")

    self.cover_conv1 = nn.Conv2d(3,8, kernel_size=3, padding="same")
    self.cover_conv2 = nn.Conv2d(8,16, kernel_size=3, padding="same")
    self.cover_conv3 = nn.Conv2d(16,32, kernel_size=3, padding="same")

    self.conv1 = nn.Conv2d(64,64, kernel_size=3, padding="same")
    self.conv2 = nn.Conv2d(64,128, kernel_size=3, padding="same")
    self.conv3 = nn.Conv2d(128,128, kernel_size=3, padding="same")
    self.conv4 = nn.Conv2d(128,64, kernel_size=3, padding="same")
    self.conv5 = nn.Conv2d(64,32, kernel_size=3, padding="same")
    self.conv6 = nn.Conv2d(32,16, kernel_size=3, padding="same")
    self.conv7 = nn.Conv2d(16,8, kernel_size=3, padding="same")
    self.conv8 = nn.Conv2d(8,3, kernel_size=3, padding="same")

  def forward(self,x):
    s, p = x
    print(1, s.shape)
    s = F.relu(self.cover_conv1(s))
    s = F.relu(self.cover_conv2(s))
    s = F.relu(self.cover_conv3(s))
    print(1, s.shape)
    print("-----")
    print(1,p.shape)
    p = F.relu(self.secret_conv1(p))
    p = F.relu(self.secret_conv2(p))
    p = F.relu(self.secret_conv3(p))
    print(1,p.shape)
    print("----")
    m = torch.cat((s, p), 1)
    print(1,m.shape)
    m = F.relu(self.conv1(m))
    m = F.relu(self.conv2(m))
    m = F.relu(self.conv3(m))
    m = F.relu(self.conv4(m))
    m = F.relu(self.conv5(m))
    m = F.relu(self.conv6(m))
    m = F.relu(self.conv7(m))
    m = F.relu(self.conv8(m))
    print(1,m.shape)
    return m

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder,self).__init__()
    self.conv1 = nn.Conv2d(3,8, kernel_size=3, padding="same")
    self.conv2 = nn.Conv2d(8,16, kernel_size=3, padding="same")
    self.conv3 = nn.Conv2d(16,32, kernel_size=3, padding="same")
    self.conv4 = nn.Conv2d(32,64, kernel_size=3, padding="same")
    self.conv5 = nn.Conv2d(64,128, kernel_size=3, padding="same")
    self.conv6 = nn.Conv2d(128,128, kernel_size=3, padding="same")
    self.conv7 = nn.Conv2d(128,64, kernel_size=3, padding="same")
    self.conv8 = nn.Conv2d(64,32, kernel_size=3, padding="same")
    self.conv8 = nn.Conv2d(64,32, kernel_size=3, padding="same")
    self.conv8 = nn.Conv2d(64,32, kernel_size=3, padding="same")
    self.conv8 = nn.Conv2d(64,32, kernel_size=3, padding="same")


  
# print(summary(Encoder(),  input_size=(1,3,256,256), device='cpu'))


