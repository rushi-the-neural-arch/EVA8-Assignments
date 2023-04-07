
# Fully Convolutional Vision Transformer

This repository contains the code for Assignment 10 of the EVA8 course. The goal of this assignment is to train a fully convolutional Vision Transformer on the CIFAR10 dataset for 24 epochs and 10 GradCAM outputs on any misclassified images.

## Data

The CIFAR10 dataset was imported using `torchvision.datasets` and augmented using the [albumentations](https://albumentations.ai) library. The following augmentations were applied on the data:
* RandomResizedCrop(32, scale=(0.75, 1
.0), ratio=(1.0, 1.0)),
* RandomHorizontalFlip(p=0.5),
* RandAugment(num_ops=1, magnitude=8),
* ColorJitter(0.1, 0.1, 0.1),
* ToTensor(),
* Normalize(cifar10_mean, cifar10_std),
* RandomErasing(p=0.25)


## Model
The `model.py` contains the Vision Transformer architecture which consists of only convolutional operations.

## Training Technique

This model was trained for 24 epochs using the One-Cycle Policy with the Adam optimizer. The One Cycle Policy, introduced by Leslie Smith in 2018, is a learning rate scheduling technique that aims to achieve better accuracy in less time. It involves a cyclic increase and decrease of the learning rate during training, where the learning rate starts low, then gradually increases until it reaches a maximum value, and then decreases back to the starting value. This helps the model to converge faster and reach a better generalization performance.
The learning rate at which the loss was the lowest was taken as the max_lr while the min_lr was taken to be one-tenth of it. No annihilation was applied at the end.

## Results
Max Train Acc - **72.87%**
\
Max Test Acc - **75.26%**
\
\
**Note**: The fact that train acc is still less than test acc proves that the model has not yet reached its full potential and is capable of reaching accuracies comparable to SOTA.

## GradCam Results
![gradcam_results](https://user-images.githubusercontent.com/34182074/230546793-fb77ca7b-f1a9-42a4-96f4-9fc1f9098d91.png)
