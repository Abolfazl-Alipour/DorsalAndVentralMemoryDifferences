# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:11:53 2019

@author: aalipour
"""

import torch
import torch.optim as optim
from echotorch.datasets.NARMADataset import NARMADataset
import echotorch.nn as etnn
import echotorch.utils
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import numpy as np
import mdp
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pdb
# Parameters
spectral_radius = 0.9
leaky_rate = 1.0
learning_rate = 0.0005
firstLayerSize = 5
n_hidden = 20
n_iterations = 1
train_sample_length = 5000
test_sample_length = 1000
n_train_samples = 2
n_test_samples = 1
batch_size = 4
momentum = 0.95
weight_decay = 0



# Use CUDA?
use_cuda = True
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed
mdp.numx.random.seed(1)
np.random.seed(2)
torch.manual_seed(1)

# NARMA30 dataset
narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10, seed=1)
narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10, seed=10)

# Data loader
trainloader = DataLoader(narma10_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(narma10_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
def param_printer(layer):
 for param in layer.parameters():
  print(param) 

#cortical column

class column(nn.Module):
    """
    cortical column model
    """
    def __init__(self):
        super(column,self).__init__()
        self.fc1=nn.Linear(in_features=1,out_features=firstLayerSize,bias=True)
        self.echo=etnn.LiESNCell(1,False,firstLayerSize, n_hidden, spectral_radius=0.9,seed=123456789)
        self.out=etnn.RRCell(n_hidden,1)
    def forward(self,x,y=None): #implement the forward pass        
        x=F.tanh(self.fc1(x))
        x=self.echo.forward(x)
        if y is not None:
            tmp=self.out(x,y)
            self.out.finalize()
            x=self.out(x)
            return x
        else:
            return self.out(x, y)
        
 # end class definition  

 
c1=column()

if use_cuda:
    c1.cuda()     
 # end if       

# Objective function
criterion = nn.MSELoss()
# Stochastic Gradient Descent
optimizer = optim.Adam(c1.parameters())#, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


# For each iteration
for epoch in range(n_iterations):
    # Iterate over batches
    i=1
    for data in trainloader:
        # Inputs and outputs
        inputs, targets = data
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
    
        # Gradients to zero
        optimizer.zero_grad()
    
        # Forward
        pdb.set_trace()
         
        out = c1(inputs,targets)
        loss = criterion(out, targets)


        loss.backward(retain_graph=True)

#    
        # Optimize
        optimizer.step()
        # Print error measures
        print(u"Train MSE: {}".format(float(loss.data)))
        print(u"Train NRMSE: {}".format(echotorch.utils.nrmse(out.data, targets.data)))
        print('conv10 parameters')
        param_printer(c1.fc1)
        print(c1.out.w_out)
        i=i+1
        c1.out.reset()
        # end for
    
    # Test reservoir
    dataiter = iter(testloader)
    test_u, test_y = dataiter.next()
    test_u, test_y = Variable(test_u), Variable(test_y)
    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
    y_predicted = c1(test_u)

    # Print error measures
    print(u"Test MSE: {}".format(echotorch.utils.mse(y_predicted.data, test_y.data)))
    print(u"Test NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, test_y.data)))
    print(u"")
    # end for
    
    
#Train MSE: 0.009042979218065739
#Train NRMSE: 0.8747499312961984
#Test MSE: 0.009726136922836304
#Test NRMSE: 0.8986288447417873
