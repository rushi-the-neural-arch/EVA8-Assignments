# Assignment 5 - Batch Normalisation and Regularisaton

This folder contains the solution to the Assignment 5 which is based on the comparison between different types of normalisation.

**model.py** - Contains the code for same models using different normalisation techniques comprising Batchnorm, Groupnorm and Layernorm.

**train.ipynb** - Contains the code required to train the above models and compare them on the basis of loss and accuracy curves.

## Explanation of different normalisation techniques

Batch normalization, group normalization, and layer normalization are all normalization techniques used in deep neural networks to improve training speed and accuracy. However, they differ in their approach to normalizing the inputs to the activation functions.

Here's a brief overview of each technique:

Batch Normalization: Batch normalization normalizes the activations of a layer for each mini-batch during training. Specifically, it subtracts the mean and divides by the standard deviation of the activations in the batch. This helps to reduce internal covariate shift and allows for more stable gradients during backpropagation.

Layer Normalization: Layer normalization normalizes the activations of a layer across all units within a given layer. Specifically, it subtracts the mean and divides by the standard deviation of the activations across all units in the layer. This helps to reduce the dependence of gradients on the scale of the parameters and allows for more stable gradients during backpropagation.

Group Normalization: Group normalization is similar to batch normalization, but instead of normalizing the activations across a batch, it normalizes them across groups of channels. The number of groups is typically set to be smaller than the number of channels. This helps to reduce internal covariate shift and allows for more stable gradients during backpropagation.

The key difference between these techniques is the scope of the normalization. Batch normalization normalizes over the batch dimension, layer normalization normalizes over the feature dimension, and group normalization normalizes over a combination of the batch and feature dimensions. This makes them suitable for different scenarios, depending on the nature of the data and the architecture of the network.
