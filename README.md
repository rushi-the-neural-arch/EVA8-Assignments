# TSAI Assignments on Benchmark Datasets

This repository contains the implementation of custom models which were trained on benchmark datasets as part of the assignments of EVA-8 Course of [The School of AI](https://theschoolof.ai).

## Data

* MNIST Dataset
* CIFAR10 dataset 

These were imported using `torchvision.datasets` and augmented using the [albumentations](https://albumentations.ai) library.

## Architectures implemented

* Custom CNN Model for Sum Prediction and Digit Prediction on MNIST Dataset
* Custom Resnet for CIFAR10
* Scratch implementation of a simple Transformer model for CIFAR10 classification
* A Simple Transformer implemented from scratch
