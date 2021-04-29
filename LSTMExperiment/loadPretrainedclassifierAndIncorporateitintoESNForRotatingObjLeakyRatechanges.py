# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:28:17 2019

@author: aalipour
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from torch.autograd import Variable
import torch.nn.functional as F
import echotorch.nn as etnn
from torch.utils.data.dataloader import DataLoader
from tomDatasetFrameSeriesAllClassesObjIdentity import tomImageFolderFrameSeriesAllClasses
import mdp
import pdb
    
class Flatten(torch.nn.Module):
  def forward(self, x):
      batch_size = x.shape[0]
      return x.view(batch_size, -1)
  
numOfClasses=16
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.frame=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5),
                     nn.ReLU(),
                     nn.MaxPool2d(2,2),
                     nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5),
                     nn.MaxPool2d(2,2),
                     Flatten(),
                     nn.Linear(in_features=12*4*4,  out_features=120),
                     nn.ReLU(),
                     nn.Linear(in_features=120, out_features=60),
                     nn.ReLU(),
                     nn.Linear(in_features=60, out_features=numOfClasses),
                     nn.ReLU())
        
#loading the pre-train network that has ~50% accuracy
network=Network()
network.frame.load_state_dict(torch.load('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/savedNetworks/objectClassifierWithTrainTestSixty2Forty48PCNTACC200Epochs'))
newnet=torch.nn.Sequential(*(list(network.frame.children()))[:-1])
newnet=torch.nn.Sequential(*(list(newnet.children()))[:-1])
            
def param_printer(layer):
 for param in layer.parameters():
  print(param) 

def printIfReqGrad(layer):
    for param in layer.parameters(): 
        print(param.requires_grad)
  
  
def freeze_layer(layer):
 for param in layer.parameters():
  param.requires_grad = False


    
def ESNRotatingObjectLrnLeakyRate(forgetRate=0.5, n_iterations=1,seed=1234567890):
    # Parameters
    spectral_radius = 0.9
    learning_rate = 0.0025
    n_hidden = 160
    batch_size = 1
    numOfClasses=16
    # numOfFrames=22
    lastLayerSize=60
    train_forgetRate=True
    inputRate=1
    train_inputRate=False
    #cortical column
    class column(nn.Module):
        """
        cortical column model
        """
        def __init__(self,preTrainedModel,forgetRate,spectral_radius = spectral_radius,n_hidden = n_hidden,numOfClasses=numOfClasses,lastLayerSize=lastLayerSize):
            super(column,self).__init__()
            self.frontEnd=preTrainedModel
            self.echo=etnn.LiESNCell(forgetRate,train_forgetRate,inputRate,train_inputRate,lastLayerSize, n_hidden, spectral_radius=spectral_radius,seed=seed)#nonlin_func=F.relu
            self.outLinear=nn.Linear(n_hidden,numOfClasses,bias=True)
        def forward(self,x,y=None,lastLayerSize=lastLayerSize): #implement the forward pass   
            # pdb.set_trace()     
            hh=np.empty((x.size(0),x.size(1),lastLayerSize))# x.siz(0) is the number of batchs and the x.size(1) is the number of frames in each class
            hh[:]=np.nan
            hh=torch.FloatTensor(hh)
            for batchNum in range(batch_size):
                m=x[batchNum,:,:,:].unsqueeze(1)
                m = self.frontEnd(m)
                hh[batchNum,:,:]=m.detach()
            x=hh 
#            pdb.set_trace()
            x=  self.echo.forward(x)
            x=  self.outLinear(x)
    
            return x
        
        # end class definition 
        
    transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.Resize((28,28)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

    train_set = tomImageFolderFrameSeriesAllClasses(root=os.path.join('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/TrainTest/revierwersAsked/syntheticObj60Train/train' ),transform=transform)
    
    test_set = tomImageFolderFrameSeriesAllClasses(root=os.path.join('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/TrainTest/revierwersAsked/syntheticObj60Train/validation' ),transform=transform)
    classes=('obj1','obj2','obj3','obj4','obj5','obj6','obj7','obj8','obj9','obj10','obj11','obj12','obj13','obj14','obj15','obj16')
    
    
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,         num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True,     num_workers=2)
    
    
    # Use CUDA?
    use_cuda = False
    use_cuda = torch.cuda.is_available() if use_cuda else False
    
    # Manual seed
    mdp.numx.random.seed(1)
    np.random.seed(2)
    torch.manual_seed(1)
 
    
     
    c1=column(newnet,forgetRate)
    
    if use_cuda:
        c1.cuda()     
     # end if       
     
     
     # Objective function
    criterion = nn.CrossEntropyLoss()
    # Adaptive Moment Estimation (Adam)
    optimizer = optim.SGD(c1.parameters(),lr=learning_rate)#, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    #freezing the pretrained front end
    freeze_layer(c1.frontEnd)
    
    # For each iteration
#    training_start_time = time.time()
    leakyRateList=[]
    accuracy=[]
    for epoch in range(n_iterations):
        for data in train_loader:
            # Inputs and outputs
            inputs, targets = data
            inputs, targets = Variable(inputs), Variable(targets)
            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        
            # Gradients to zero
            optimizer.zero_grad()
        
            # Forward
            out = c1(inputs,targets)
            loss = criterion(out.permute(0,2,1), targets.long())
    
    
            loss.backward(retain_graph=False)
    
            # Optimize
            optimizer.step()
            #end for 
            
        leakyRateList.append((c1.echo.forgetRate.detach().numpy().astype(np.float)))#/(c1.echo.inputRate.detach().numpy().astype(np.float)))
       # Test reservoir
        correct = 0
        total = 0
        for images, labels in test_loader:
            with torch.no_grad():
                images, labels = Variable(images), Variable(labels)
                if use_cuda:
                    outputs = c1(images.cuda())
                    _,predicted=torch.max(outputs[0],dim=1)
    #                showMe=predicted[1]-labels.cuda()[0]
    #                numberOfMistakes+=len([i for i, e in enumerate(showMe) if e != 0])
                    correct += (predicted == labels.cuda()).sum()
                    total += labels.size(1)
                    del labels,outputs
                    torch.cuda.empty_cache()
                else:
                     outputs = c1(images)
                     _,predicted=torch.max(outputs[0],dim=1)
                     correct += (predicted == labels).sum()
                     total += labels.size(1)
        correct=correct.cpu().numpy()
        total=(torch.tensor(total)).numpy()
        accuracy.append(correct/total)
        # end for

    return accuracy, leakyRateList

##save network parameters for the rocord
#torch.save(network.frame.state_dict(), os.path.join('E:\\', 'Abolfazl' , '2ndYearproject','code','savedNetworks','ESNForRotatingObj'))
#
#
##function for ploting convolutional kernels
#def plot_kernels(tensor, num_cols=6):
#    num_kernels = tensor.shape[0]
#    num_rows = num_kernels // num_cols
#    fig = plt.figure(figsize=(num_cols,num_rows))
#    for i in range(num_kernels):
#        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
#        ax1.imshow(tensor[i][0,:,:], cmap='gray')
#        ax1.axis('off')
#        ax1.set_xticklabels([])
#        ax1.set_yticklabels([])
##ploting the first conv2d layer's kernels
#plot_kernels(list(c1.frontEnd.state_dict().values())[0])
#
#
##plot output layer
#plt.imshow(c1.outLinear.weight.detach())
    



