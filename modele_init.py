#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:25:01 2019

@author: anasahouzi
"""
import torch



# Creation of a simple neural network architecture having 5 nodes in the input
# layer, 3 in the hidden layer, and one in the output layer.

## sigmoid activation function using pytorch
def sigmoid_activation(z):
    return 1 / (1 + torch.exp(-z))

## function to calculate the derivative of activation
def sigmoid_delta(x):
  return x * (1 - x)



n_input, n_hidden, n_output = 5, 3, 1

# Parameter intialisation: we initialise the weights and bias as tensor variables,
# Tensors are the base data structures of PyTorch, and they are used for 
# building different types of neural network. They are the generalisation of 
# arrays and matrices, in other words tensors are N-dimensional matrices

## initialize tensor for inputs, and outputs 
x = torch.randn((1, n_input))
y = torch.randn((1, n_output))

## initialize tensor variables for weights 
w1 = torch.randn(n_input, n_hidden) # weight for hidden layer
w2 = torch.randn(n_hidden, n_output) # weight for output layer

## initialize tensor variables for bias terms 
b1 = torch.randn((1, n_hidden)) # bias for hidden layer
b2 = torch.randn((1, n_output)) # bias for output layer


## activation of hidden layer 
z1 = torch.mm(x, w1) + b1
a1 = sigmoid_activation(z1)


## activation (output) of final layer 
z2 = torch.mm(a1, w2) + b2
output = sigmoid_activation(z2)


# Loss computation
loss = y - output


## compute derivative of error terms
delta_output = sigmoid_delta(output)
delta_hidden = sigmoid_delta(a1)


## backpass the changes to previous layers 
d_outp = loss * delta_output
loss_h = torch.mm(d_outp, w2.t())
d_hidn = loss_h * delta_hidden


# Updating the Parameters: Finally, the weights and bias are updated 
# using the delta changes received from the above backpropagation step.


learning_rate = 0.1
w2 += torch.mm(a1.t(), d_outp) * learning_rate
w1 += torch.mm(x.t(), d_hidn) * learning_rate
b2 += d_outp.sum() * learning_rate
b1 += d_hidn.sum() * learning_rate






































