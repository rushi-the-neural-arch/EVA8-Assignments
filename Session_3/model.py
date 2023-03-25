import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, p=0):
        super(Net, self).__init__()
        self.p = p
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.dil_conv1 = nn.Conv2d(32, 32, 5, dilation=4)
        self.drop1 = nn.Dropout(p=self.p)
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.dil_conv2 = nn.Conv2d(64, 64, 3, dilation=4)
        self.drop2 = nn.Dropout(p=self.p)
        self.conv3 = nn.Conv2d(64, 64, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.dil_conv3 = nn.Conv2d(64, 64, 3, dilation=2)
        self.drop3 = nn.Dropout(p=self.p)
        self.conv4 = nn.Conv2d(64, 64, 3, padding='same')
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding='same', groups=64)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, 3, padding='same', groups=128)
        self.gap = nn.AvgPool2d(4)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.drop1(self.dil_conv1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.dil_conv2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.dil_conv3(F.relu(self.bn3(self.conv3(x)))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.gap(self.conv6(x))
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
    
net = Net(p=0.3)
from torchsummary import summary
print(summary(net, (3, 32, 32)))
