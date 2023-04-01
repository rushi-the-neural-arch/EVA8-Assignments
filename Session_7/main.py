import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from data import AlbumentationImageDataset
from model import ResNet18
from train_test import fit_model
from utils import get_misclassified_images, plot_imgs, get_cam, plot_cam


SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available:", cuda)

# For reproducibility
torch.manual_seed(SEED)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if cuda:
    torch.cuda.manual_seed(SEED)
    BATCH_SIZE=64
else:
    BATCH_SIZE=64


# loading the dataset
exp = datasets.CIFAR10('./data', train=True, download=True)
exp_data = exp.data

# Calculate the mean and std for normalization
print('[Train]')
print(' - Numpy Shape:', exp_data.shape)
print(' - min:', np.min(exp_data, axis=(0,1,2)) / 255.)
print(' - max:', np.max(exp_data, axis=(0,1,2)) / 255.)
print(' - mean:', np.mean(exp_data, axis=(0,1,2)) / 255.)
print(' - std:', np.std(exp_data, axis=(0,1,2)) / 255.)
print(' - var:', np.var(exp_data, axis=(0,1,2)) / 255.)


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True )
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True)


train_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(trainset, train=True), batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(testset, train=False), batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=1)


net = ResNet18().to(device)

import copy
train_net_1 = copy.deepcopy(net).to(device)
optimizer = torch.optim.SGD(train_net_1.parameters(), lr=0.003, momentum=0.9)
train_net_1, history = fit_model(
    train_net_1, device=device,
    criterion = nn.CrossEntropyLoss(),
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,  
    NUM_EPOCHS=20
)

misclassified_imgs = get_misclassified_images(train_net_1, test_loader, device)
plot_imgs(exp.classes, misclassified_imgs)

# Get the keys(image indices) of the misclassified images.
keys_list = iter(list(misclassified_imgs.keys()))
     

# Get the layer name of the ResNet18 model, that would be used to get the gradients required
# to compute the class activation maps.

target_layers = [train_net_1.layer2[1]]
cam = get_cam(train_net_1, target_layers, use_cuda = True)
     
# Use plot_cam() function to plot the misclassified images.

plot_cam(cam, misclassified_imgs, exp.classes, keys_list=keys_list)