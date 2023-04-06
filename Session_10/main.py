import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision
from torchsummary import summary
import copy
from torch_lr_finder import LRFinder
import time

from model import ViT
from utils import get_cam, plot_cam, plot_imgs, get_misclassified_images

DATA_DIR='./data'

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", DEVICE)



net = ViT(image_size=32, patch_size=2, num_classes=10, dim=32, depth=4, heads=4, mlp_dim=128, pool = 'cls', channels = 3, dim_head = 32, dropout = 0., emb_dropout = 0.)
net = net.to(DEVICE)
summary(net, (3, 32, 32))

IMAGE_SIZE = 32

NUM_CLASSES = 10
NUM_WORKERS = 2
BATCH_SIZE = 16
EPOCHS = 25

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-1

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=1, magnitude=8),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
    transforms.RandomErasing(p=0.25)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=NUM_WORKERS)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=NUM_WORKERS)


exp_net = copy.deepcopy(net).to(DEVICE)
optimizer = torch.optim.AdamW(exp_net.parameters(), lr=0.0001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
lr_finder = LRFinder(exp_net, optimizer, criterion, device=DEVICE)
lr_finder.range_test(trainloader, end_lr=10, num_iter=200)
print(lr_finder.history['loss'])
lr_finder.plot()
min_loss = min(lr_finder.history['loss'])
ler_rate_1 = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
print("Max LR is {}".format(ler_rate_1))

exp_net = copy.deepcopy(net).to(DEVICE)
optimizer = torch.optim.Adam(exp_net.parameters(), lr=ler_rate_1/10)
criterion = nn.CrossEntropyLoss()
lr_finder = LRFinder(exp_net, optimizer, criterion, device=DEVICE)
lr_finder.range_test(trainloader, end_lr=ler_rate_1*10, num_iter=200)
lr_finder.plot()
min_loss = min(lr_finder.history['loss'])
ler_rate_2 = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
print("Max LR is {}".format(ler_rate_2))


ler_rate = ler_rate_2
print("Determined Max LR is:", ler_rate)


clip_norm = True
lr_schedule = lambda t: np.interp([t], [0, EPOCHS*2//5, EPOCHS*4//5, EPOCHS], 
                                  [0, ler_rate, ler_rate/20.0, 0])[0]

model = copy.deepcopy(net).to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0
    for i, (X, y) in enumerate(trainloader):
        model.train()
        X, y = X.cuda(), y.cuda()

        lr = lr_schedule(epoch + (i + 1)/len(trainloader))
        opt.param_groups[0].update(lr=lr)

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(X)
            loss = criterion(output, y)

        scaler.scale(loss).backward()
        if clip_norm:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
        
    model.eval()
    test_acc, m = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)

    print(f'Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Test Acc: {test_acc/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')


classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

misclassified_imgs = get_misclassified_images(model, testloader, DEVICE)
plot_imgs(classes, misclassified_imgs)

# Get the keys(image indices) of the misclassified images.
keys_list = iter(list(misclassified_imgs.keys()))
     

# Get the layer name of the ResNet18 model, that would be used to get the gradients required
# to compute the class activation maps.

target_layers = [model.transformer.layers[2][1]]
cam = get_cam(model, target_layers, use_cuda = True)

# Use plot_cam() function to plot the misclassified images.

plot_cam(cam, misclassified_imgs, classes, keys_list=keys_list)
