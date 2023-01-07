
# Sum Prediction and Digit Prediction on MNIST Dataset

This notebook can be used to train a neural network which, when given an MNIST image and a randomly generated number, would predict the number in the image and also the sum of both the numbers.


## Random Number Generation Strategy

A PyTorch function, `torch.randint`, has been used to generate the random number.


## Combination of the 2 inputs

The first input is an image. So, it was first converted into an embedding of length 2048. Then, the second input, the random number, was concatenated to it. This is followed by 2 different sets of fully connected layers. One is for the digit prediction while the other is for sum prediction.


## Loss function used

PyTorch's Cross Entropy function has been used as both fully connected layers output a vector of a particular length (10 for digit prediction and 19 for sum prediction). There is no softmax layer at the end as this cross entropy function applies a softmax layer to its inputs by default.


## Evaluation

Accuracy has been used to measure the performance of the model when given unseen data.


## Results

Accuracy on test set:

Digit Recognition - 99.25

Sum Prediction - 93.9


