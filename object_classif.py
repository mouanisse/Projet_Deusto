#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matlab.pyplot as plt
from IPython.display import HTML, display
import sys



print(sys.version)

# make sure to enable GPU acceleration!
device = 'cuda'

# Installing PyTorch
# !pip3 install torch torchvision

# Check if PyTorch is installed
print('PyTorch version:', torch.__version__)



class ProgressMonitor(object):
    """
    Custom IPython progress bar for training
    """
    
    tmpl = """
        <p>Loss: {loss:0.4f}   {value} / {length}</p>
        <progress value='{value}' max='{length}', style='width: 100%'>{value}</progress>
    """

    def __init__(self, length):
        self.length = length
        self.count = 0
        self.display = display(self.html(0, 0), display_id=True)
        
    def html(self, count, loss):
        return HTML(self.tmpl.format(length=self.length, value=count, loss=loss))
        
    def update(self, count, loss):
        self.count += count
        self.display.update(self.html(self.count, loss))



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
train_set = CIFAR10('data', train=True, download=True, transform=_tasks)
valid_set = CIFAR10('data', train=False, download=True, transform=_tasks)
print(train_set.shape)
print(valid_set.shape)



## create iterator objects for train and valid datasets
trainloader = DataLoader(train_set, batch_size=256, num_workers=0, shuffle=True)
validloader = DataLoader(valid_set, batch_size=256, num_workers=0, shuffle=True)


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
        
        # x.reshape(-1, 1024) is equivalent to flatten in TensorFlow
        x = x.reshape(-1, 1024) ## reshaping 
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    

# Create the model
model = Model()
# Move the model (memory and operations) to the CUDA device
model.to(device)


# Define the loss function and the optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, 
                      momentum = 0.9, nesterov = True)



# We are now ready to train the model. The core steps will remain the same as
# we saw earlier: Forward Propagation, Loss Computation, Backpropagation, 
# and updating the parameters.

for epoch in range(1, 31): ## run the model for 30 epochs
    
    # Those two vectors will contain the loss for every batch,
    # and the loss that will be computed next to the bar progress represents
    # the mean of every vector.
    train_loss, valid_loss = [], []
    
    ## We r setting the mode of training, it's like we r telling our nn, to 
    ## turn the training mode "on".
    model.train()
    
    # create a progress bar
    progress = ProgressMonitor(length=len(train_set))
    
    
    for data, target in trainloader:
        
        # Move the training data, and labels to the GPU.
        data = data.to(device)
        target = target.to(device)
        
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
        
        ## 5. update progress bar
        progress.update(data.shape[0], loss)
        
        ## 6. The loss value of each batch is appended train_loss
        train_loss.append(loss.item())
        
    ## We r now telling our nn that this is testing part, and that the 
    ## training part is over, we can also use model.train(mode=False).
    model.eval()
    
    y_pred = []
    
    for data, target in validloader:
        
        # Move the validation data, and labels to the GPU.
        data = data.to(device)
        target = target.to(device)
        
        ## 1. Forward propagation
        output = model(data)
        
        ## 2. Loss calculation
        loss = loss_function(output, target)
        
        ## 3. The loss value of each batch is appended valid_loss
        valid_loss.append(loss.item())
        
        ## 4. save predictions
        ## extends the list by adding all items of a list 
        ## (passed as an argument) to the end.
        y_pred.extend(output.argmax(dim=1).cpu().numpy())
        
        # Calculate validation accuracy
        y_pred = torch.tensor(y_pred, dtype=torch.int64)
        val_accuracy = torch.mean((y_pred == valid_set.test_labels).float())
        print('Validation accuracy: {:.4f}%'.format(float(val_accuracy) * 100))
        
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), 
           "Valid Loss: ", np.mean(valid_loss))
    

## dataloader for validation dataset 
dataiter = iter(validloader)

# we r using iter(), so we can loop for all the tuples in dataiter 
# using next()
data, labels = dataiter.next()
output = model(data)


# Generate the predictions on the validation set
# Functions with torch.xxx() operate on tensors only !!!!!!

_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())
print ("Actual:", labels[:10])
print ("Predicted:", preds[:10])
    





























