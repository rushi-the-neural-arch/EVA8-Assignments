import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Res_Block(nn.Module):
  def __init__(self, in_channels):
    super(Res_Block, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(in_channels)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(x)))
    return out + x


class Custom_Resnet(nn.Module):
  def __init__(self):
    super(Custom_Resnet, self).__init__()
    self.prep_conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)      # Output - 64x32x32
    self.prep_bn = nn.BatchNorm2d(64)

    self.conv1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)      # Output - 128x32x32
    self.pool1 = nn.MaxPool2d(2)      # Output - 128x16x16
    self.bn1 = nn.BatchNorm2d(128)
    self.res1 = self.res_block(128)      # Output - 128x16x16

    self.conv2 = nn.Conv2d(128, 256, 3, padding=1, bias=False)      # Output - 256x16x16
    self.pool2 = nn.MaxPool2d(2)      # Output - 256x8x8
    self.bn2 = nn.BatchNorm2d(256)

    self.conv3 = nn.Conv2d(256, 512, 3, padding=1, bias=False)      # Output - 512x8x8
    self.pool3 = nn.MaxPool2d(2)      # Output - 512x4x4
    self.bn3 = nn.BatchNorm2d(512)
    self.res2 = self.res_block(512)      # Output - 512x4x4

    self.pool4 = nn.MaxPool2d(4)      # Output - 512x1x1

    self.fc = nn.Linear(512, 10)

  def forward(self, x):
    x = F.relu(self.prep_bn(self.prep_conv(x)))
    x = F.relu(self.bn1(self.pool1(self.conv1(x))))
    x = self.res1(x) + x
    
    x = F.relu(self.bn2(self.pool2(self.conv2(x))))

    x = F.relu(self.bn3(self.pool3(self.conv3(x))))
    x = self.res2(x) + x

    x = self.pool4(x)
    x = x.reshape(-1, 512)

    x = self.fc(x)
    x = x.view(-1, 10)   
    return F.log_softmax(x, dim=-1)

  def res_block(self, in_channels):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU()
    )
    return layer
