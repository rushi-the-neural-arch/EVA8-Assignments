
# GradCAM Visualization of Resnet 18 on CIFAR10

This repository contains the code for Assignment 7 of the EVA6 course. The goal of this assignment is to train ResNet18 on the CIFAR10 dataset for 20 epochs and display loss curves for test and train datasets, a gallery of 10 misclassified images, and 10 GradCAM outputs on any misclassified images. The code follows the structure mentioned in the assignment instructions, with separate files for models, main, and utilities.

## Data

The CIFAR10 dataset was imported using `torchvision.datasets` and augmented using the [albumentations](https://albumentations.ai) library. The following augmentations were applied on the data:
* RandomCrop(32, padding=4)
* CutOut(16x16)



## Model
The `model.py` contains the ResNet18 model architecture with ResNet18 and ResNet34 models.

## Training

The model was trained using stochastic gradient descent with a learning rate of 0.003 and momentum of 0.9 for 20 epochs.


## Results
Max Train Acc - **86.68%**
Max Test Acc - **83.77%**

## GradCAM Images
![gradcam](https://user-images.githubusercontent.com/34182074/229306685-cc366104-085d-42c1-9343-ac5cf779fd23.png)
