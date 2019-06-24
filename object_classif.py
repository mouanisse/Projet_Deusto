#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.nn as nn
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



# To perform object classification, we will be using CIFAR-10 dataset,
# that is included in the torchvision package.
## load the dataset 
cifar = CIFAR10('data', train=True, download=True, transform=_tasks)


split = int(0.8*len(cifar))
idx = list(range(len(cifar)))
train_idx, val_idx = idx[:split], idx[split:]

## create training and validation sampler objects
# Samples elements randomly from a given !!!!!!!list!!!!!! of indices, 
# without replacement. 
# This variable just stores the list of possible indices that you provided. 
# When the sampler is going to be used, the indices it returns will 
# be randomized.
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)



## create iterator objects for train and valid datasets
trainloader = DataLoader(cifar, batch_size=256, sampler=tr_sampler)
validloader = DataLoader(cifar, batch_size=256, sampler=val_sampler)


# Define the neural network model

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        ## define the feature extraction layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # define the classification layers, "nn.Linear" is the fully connected
        # layer
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # x.view(-1, 1024) is equivalent to flatten in TensorFlow
        x = x.view(-1, 1024) ## reshaping 
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    

model = Model()

# Define the loss function and the optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, 
                      momentum = 0.9, nesterov = True)



# We are now ready to train the model. The core steps will remain the same as
# we saw earlier: Forward Propagation, Loss Computation, Backpropagation, 
# and updating the parameters.

for epoch in range(1, 31): ## run the model for 30 epochs
    train_loss, valid_loss = [], []
    
    ## We r setting the mode of training, it's like we r telling our nn, to 
    ## turn the training mode "on".
    model.train()
    
    
    for data, target in trainloader:
        
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
    

## dataloader for validation dataset 
dataiter = iter(validloader)
data, labels = dataiter.next()
output = model(data)


# Generate the predictions on the validation set
# Functions with torch.xxx() operate on tensors only !!!!!!

_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())
print ("Actual:", labels[:10])
print ("Predicted:", preds[:10])
    





























