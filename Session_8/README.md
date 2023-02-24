
# Custom ResNet architecture for CIFAR10

This repository contains the implementation of a custom **ResNet** architecture for the CIFAR10 dataset.

## Data

The CIFAR10 dataset was imported using `torchvision.datasets` and augmented using the [albumentations](https://albumentations.ai) library. The following augmentations were applied on the data:
* RandomCrop of 32, 32 (after padding of 4)
* FlipLR 
* CutOut(8, 8)

## Architecture

**PrepLayer** - 32x32x3 to 32x32x64
* Conv 3x3 (s1, p1) >> BN >> RELU

**Layer1** - 32x32x64 to 16x16x128
* X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU
* R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X)
* Add(X, R1)

**Layer 2** - 16x16x128 to 8x8x256
* Conv 3x3
* MaxPooling2D
* BN
* ReLU

**Layer 3** - 8x8x256 to a vector of size 10
* X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU
* R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X)
* Add(X, R2)
* MaxPooling with Kernel Size 4
* FC Layer
* SoftMax



## Training Technique

This model was trained for 24 epochs using the One-Cycle Policy. The One Cycle Policy, introduced by Leslie Smith in 2018, is a learning rate scheduling technique that aims to achieve better accuracy in less time. It involves a cyclic increase and decrease of the learning rate during training, where the learning rate starts low, then gradually increases until it reaches a maximum value, and then decreases back to the starting value. This helps the model to converge faster and reach a better generalization performance.

The learning rate at which the loss was the lowest was taken as the `max_lr` while the `min_lr` was taken to be one-tenth of it. Also, maximum learning rate was reached within 5 epochs. No annihilation was applied at the end.

## Results

Final Training Accuracy - **95.18 %**
\
Final Testing Accuracy - **89.01 %**
