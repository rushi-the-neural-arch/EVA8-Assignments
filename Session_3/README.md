
# Custom Light Architecture for CIFAR10

This repository contains the implementation of a custom ResNet architecture for the CIFAR10 dataset.

## Data

The CIFAR10 dataset was imported using `torchvision.datasets` and augmented using the [albumentations](https://albumentations.ai) library. The following augmentations were applied on the data:
* HorizontalFlip
* ShiftScaleRotate
* CoarseDropout


## Model
The motive was to create a convolution neural net without the use of max pooling. So, dilated convolutions were used to halve the image size. At the beginning, normal convolutions were used to capture the important features of the input image while the tail of the model involves depthwise separable convolutions. Finally, a net ends with a global average pooling layer followed by a fully connected layer to output a 10 dimensional vector.
\
\
Total Parameters - 200K


## Training

The model was trained using stochastic gradient descent with a learning rate of 0.01 and momentum of 0.9 for 100 epochs. The maximum testset accuracy achieved was 76%.
