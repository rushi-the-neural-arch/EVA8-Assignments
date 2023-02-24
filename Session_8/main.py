import torch
import torchvision
import numpy as np
import copy
import torch.nn as nn
import albumentations as A
from torchvision import datasets
from train import fit_model
from data import AlbumentationImageDataset
from Assignment_8_model import Custom_Resnet
from torch_lr_finder import LRFinder
from torchsummary import summary



SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available:", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    BATCH_SIZE=512
else:
    BATCH_SIZE=32


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True )
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True)


train_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(trainset, train=True), batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(testset, train=False), batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
net = Custom_Resnet().to(device)
print(summary(net, input_size=(3, 32, 32)))


found_lrs = []

for i in range(5):
  exp_net = copy.deepcopy(net).to(device)
  optimizer = torch.optim.SGD(exp_net.parameters(), lr=0.001, momentum=0.9)
  criterion = nn.NLLLoss()
  lr_finder = LRFinder(exp_net, optimizer, criterion, device=device)
  lr_finder.range_test(train_loader, end_lr=2, num_iter=200)
  lr_finder.plot()
  min_loss = min(lr_finder.history['loss'])
  ler_rate = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
  print("Max LR is {}".format(ler_rate))
  found_lrs.append(ler_rate)

ler_rate = min(found_lrs)
print("Determined min LR is:", ler_rate)

train_net_1 = copy.deepcopy(net).to(device)
optimizer = torch.optim.SGD(train_net_1.parameters(), lr=(ler_rate/10), momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=ler_rate,
                                                steps_per_epoch=len(train_loader), 
                                                epochs=24,
                                                pct_start=(4/24),
                                                div_factor=10,
                                                three_phase=False, 
                                                final_div_factor=50,
                                                anneal_strategy='linear'
                                                )

train_net_1, history = fit_model(
    train_net_1, device=device,
    criterion = nn.CrossEntropyLoss(),
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer, 
    scheduler=scheduler, 
    NUM_EPOCHS=24
)
