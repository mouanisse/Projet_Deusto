#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:32:41 2019

@author: anasahouzi
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# With module F, we can apply the activation functions
import torch.nn.functional as F
from torch import optim
import numpy as np



# Data preprocessing: We need to transform the raw dataset into tensors and 
# normalize them in a fixed range. The torchvision package provides a utility 
# called transforms which can be used to combine different transformations
# together.

_tasks = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


# The first transformation converts the raw data into tensor variables and 
# the second transformation performs normalization using the below operation:
# x_normalized = x-mean / std. The first vector is the mean vector for each 
# dimension (R, G, B), and the second one is the standard deviation.



## Load MNIST Dataset and apply transformations
mnist = MNIST("data", download=True, train=True, transform=_tasks)

print("hola1")


# Another excellent utility of PyTorch is DataLoader iterators which provide 
# the ability to batch, shuffle and load the data in parallel using 
# multiprocessing workers.


## create training and validation split 
split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]

print("hola2")


## create sampler objects using SubsetRandomSampler
# Samples elements randomly from a given !!!!!!!list!!!!!! of indices, 
# without replacement.
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

print("hola3")


## create iterator objects for train and valid datasets
# Data loader. Combines a dataset and a sampler, and provides single
# or multi-process iterators over the dataset.
trainloader = DataLoader(mnist, batch_size=256, sampler=tr_sampler)
validloader = DataLoader(mnist, batch_size=256, sampler=val_sampler)

print(trainloader)

print("hola4")

# The neural network architectures in PyTorch can be defined in a class which 
# inherits the properties from the base class from nn package called Module. 
# This inheritance from the nn.Module class allows us to implement, access, and 
# call a number of methods easily. We can define all the layers inside the 
# constructor of the class, and the forward propagation steps inside 
# the forward function.


class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 128)
        self.output = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.hidden(x)
        x = F.sigmoid(x)
        x = self.output(x)
        return x
    
        
  
    
    
model = Model()

# Define the loss function and the optimizer using the nn and optim package:
loss_function = nn.CrossEntropyLoss()


# model.parameters() contains the learnable parameters (weights, biases) of
# a torch model, that will be updated using the SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, 
                      momentum = 0.9, nesterov = True)

# We are now ready to train the model. The core steps will remain the same as
# we saw earlier: Forward Propagation, Loss Computation, Backpropagation, 
# and updating the parameters.

for epoch in range(1, 11): ## run the model for 10 epochs
    train_loss, valid_loss = [], []
    
    ## We r setting the mode of training, it's like we r telling our nn, to 
    ## turn the training mode "on".
    model.train()
    
    print("hola5")
    for data, target in trainloader:
        
        print("hola6")
        # When you start your training loop, ideally you should zero out the 
        # gradients so that you do the parameter update correctly. Else the 
        # gradient would point in some other directions than the intended 
        # direction towards the minimum.
        optimizer.zero_grad()
        
        ## 1. forward propagation
        output = model(data)
        
        ## 2. loss calculation
        loss = loss_function(output, target)
        
        ## 3. backward propagation
        loss.backward()
        
        ## 4. weight optimization
        optimizer.step()
        
        train_loss.append(loss.item())
        
    ## We r now telling our nn that this is testing part, and that the 
    ## training part is over, we can also use model.train(mode=False).
    model.eval()
    
    for data, target in validloader:
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())
        
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), 
           "Valid Loss: ", np.mean(valid_loss))
    
    
#  Once the model is trained, make the predictions on the validation data.
## dataloader for validation dataset 
    
dataiter = iter(validloader)
data, labels = dataiter.next()
output = model(data)
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())
print ("Actual:", labels[:10])
print ("Predicted:", preds[:10])































