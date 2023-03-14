# Problem Statement
In this assignment you need to implement a feedforward neural network and write the backpropagation code for training the network. We strongly recommend using numpy for all matrix/vector operations. You are not allowed to use any automatic differentiation packages. This network will be trained and
tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784
pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.

# Question 1 - Plotting each class in fashion-MNIST

# Question 2,3 - Implementation

# Question 4 - Hyperparameter Search

--------------------------------
**Strategy**

1. As grid search can be computationally intensive, I opted not to use this method. Instead, I tried random search as some articles suggested it was a good alternative, and it did indeed find some promising models. However, many of the models generated through this method had low accuracy. 

2. However, random search proved to be advantageous as it provided valuable insights into the performance of various models. Specifically, it helped me understand which models were a good fit and which ones were not, and why that was the case.

3. To find the best fit, I opted bayesian search. As bayesian sweep can incorporate prior knowledge about the hyperparameters, it can improve the efficiency of the optimization process and give me the best fit in less runs.

4. I have performed two sets of searches.

>A Random and a Bayesian sweep of ~90 runs each with the configurations given in the question

>A Bayesian search of ~70 runs with the configurations below. Based on the results of previous experiments and my own analysis, I have discarded and added certain hyper-parameters that can report low and high accuracy respectively

```
- number of epochs: 10
- number of hidden layers:  3
- size of every hidden layer:  32, 64, 128
- weight decay (L2 regularisation): 0, 0.0001, 0.0005
- learning rate: 1e-1, 1e-3
- optimizer:  sgd, momentum, nesterov, rmsprop, adam, nadam
- batch size: 16, 32, 64
- weight initialisation: Xavier
- activation functions: sigmoid, tanh, relu, leakyRelu

```

# Question 5 - Best Accuracy

# Question 6 - Parellel co-ordinates plot

# Question 7 - Confusion Matrix

# Question 8 - MSE vs CE

# Question 10 - MNIST Dataset

1. The Cross-Entropy loss function and Xavier weight initialisation were consistently the best performing options, and thus I have selected these for my model.
2. Adam and Nadam optimisers yielded the highest accuracy in 10 epochs in the majority of cases, and were therefore selected for use in my model.
3. As I mentioned above, deeper models with more hidden layers don't seem to be showing any improvement over Models with 3 hidden layers. Turns out most of the best performing models have 3 hidden layers only.
4. Hence to fine-tune the model further, I should focus on the activation function. The results of sweeps clearly indicated that ReLU and Leaky ReLU reported better accuracy than sigmoid and tanh. So, I chose 2 architectures with Leaky ReLU and one with ReLU.

Finally, the top-performing models with these parameters were selected for testing on the MNIST dataset. The three architectures selected were the ones that showed the best performance in the class of their respective activation functions.

| | Architechture 1 | Architechture 2 | Architechture 3|
|----------|----------|----------|----------|
| Epochs | 10 | 10 | 10 |
| Hidden Layers | [64,64,64] | [128,128,128] |  [128,128,128]|
| Batch Size | 32 | 32 | 32|
| Activation function | Leaky ReLU |  Leaky ReLU | ReLU |
| Optimiser | Nadam | Adam | Adam |
| Learning Rate | 1e-3 | 1e-3 | 1e-3 |
| Weight Decay | 0 | 0.0001 | 0.0001 |
| Loss | Cross Entropy | Cross Entropy | Cross Entropy |
| Weight Initialisation | Xavier | Xavier | Xavier |
| Accuracy on Fashion-MNIST | 88.88 | 88.73 | 88.15  |
| Accuracy on MNIST Test data| 96.85 | 97.42| 97.07 |






