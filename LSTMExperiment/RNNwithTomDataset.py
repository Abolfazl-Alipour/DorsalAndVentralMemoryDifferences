# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:06:08 2019

@author: aalipour
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dset
spectral_radius = 0.9
leaky_rate = 1.0
learning_rate = 0.001
firstLayerSize=12*4*4
n_hidden=30
n_iterations = 5
train_sample_length = 5000
test_sample_length = 1000
n_train_samples = 10
n_test_samples = 1
batch_size = 1
momentum = 0.95
weight_decay = 0
n_layers=2

use_cuda = True
use_cuda = torch.cuda.is_available() if use_cuda else False

transform = transforms.Compose(
[transforms.Grayscale(num_output_channels=1),
 transforms.Scale((28,28)),
 transforms.ToTensor(),
 transforms.Normalize((0.5,), (0.5,))])

train_set = dset.ImageFolder(root=os.path.join('E:', 'Abolfazl' , '2ndYearproject' , 'ImagesGeneratedTom' , 'rotatedStims' ),transform=transform)

test_set = dset.ImageFolder(root=os.path.join('E:','Abolfazl','2ndYearproject','ImagesGeneratedTom','rotatedStims'),transform=transform)

classes=('obj1','obj2','obj3','obj4','obj5','obj6','obj7','obj8','obj9','obj10','obj11','obj12','obj13','obj14','obj15','obj16')


#train_set=torchvision.datasets.FashionMNIST(
#        root= os.path.join('C','Users','MAASF','Anaconda2','envs','MyEnv,Lib','site-packages','torchvision','datasets')
#        ,train=True
#        ,download=True
#        ,transform=transforms.Compose([
#                transforms.ToTensor()
#                ])
#)
#
#test_set=torchvision.datasets.FashionMNIST(
#        root= os.path.join('C','Users','MAASF','Anaconda2','envs','MyEnv,Lib','site-packages','torchvision','datasets')
#        ,train=False
#        ,download=True
#        ,transform=transforms.Compose([
#                transforms.ToTensor()
#                ])
#)
#
#len(test_set)
#DataLoader takes in a dataset and a sampler for loading (num_workers deals with system level memory) 
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,shuffle=True,         num_workers=2)

    return(train_loader)

batch_size=4
def get_test_loader(batch_size):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,shuffle=True,     num_workers=2)
    return(test_loader)
    
## model

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        
        self.rnn=nn.RNN(firstLayerSize, n_hidden, n_layers, batch_first=True)
        self.out=nn.Linear(in_features=n_hidden,out_features=16,bias=True)
    def forward(self,x,h): #implement the forward pass        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x,hn=self.rnn(x,h)
        x=F.tanh(self.out(x))
        return x,hn
    





network=Network()
print(network)
network.cuda()




##### train loop

def createLossAndOptimizer(net, learning_rate=0.001):
    
    #Loss function
    loss = nn.CrossEntropyLoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)

learning_rate=0.001
loss, optimizer = createLossAndOptimizer(network, learning_rate)
# Objective function
criterion = nn.MSELoss()
# Stochastic Gradient Descent
optimizer = optim.Adam(network.parameters())#, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)



def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_iterations):
    # Iterate over batches
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        h=torch.randn(n_layers,batch_size,n_hidden).cuda()
        if use_cuda:
            h.cuda()
        for i, data in enumerate(train_loader, 0):
            # Inputs and outputs
            inputs, targets = data
            inputs, targets = Variable(inputs), Variable(targets)
            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        
            # Gradients to zero
            optimizer.zero_grad()
        
            # Forward
            #pdb.set_trace()
            out,h = network(inputs,h)
            loss = criterion(out, targets)
        
            # Backward pass
            loss.backward(retain_graph=True)
        
            # Optimize
            optimizer.step()
            h.detach()
            # end for        
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
# =============================================================================
#         total_val_loss = 0
#         for inputs, labels in val_loader:
#             
#             #Wrap tensors in Variables
#             inputs, labels = Variable(inputs), Variable(labels)
#             
#             #Forward pass
#             val_outputs = net(inputs)
#             val_loss_size = loss(val_outputs, labels)
#             total_val_loss += val_loss_size.data[0]
#             
#         print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
# =============================================================================
        
        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    


trainNet(network, batch_size=4, n_epochs=5, learning_rate=0.001)

network.eval()
correct = 0
total = 0
test_loader=get_test_loader(batch_size)
for images, labels in test_loader :
    images = Variable(images.float())
    outputs = network(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))


