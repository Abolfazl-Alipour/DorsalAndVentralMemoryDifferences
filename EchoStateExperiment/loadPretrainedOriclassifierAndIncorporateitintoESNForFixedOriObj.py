# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:00:30 2019

@author: aalipour
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:42:02 2019

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
from tomDatasetFrameSeriesAllClassesFixedOri import tomImageFolderFrameSeriesAllClasses
import mdp
import pdb

class Flatten(torch.nn.Module):
  def forward(self, x):
      batch_size = x.shape[0]
      return x.view(batch_size, -1)
  
def param_printer(layer):
 for param in layer.parameters():
  print(param) 


def printIfReqGrad(layer):
    for param in layer.parameters(): 
        print(param.requires_grad)
  
  
def freeze_layer(layer):
 for param in layer.parameters():
  param.requires_grad = False

  
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
                     nn.Linear(in_features=60, out_features=16),
                     nn.ReLU())
 
    

network=Network()
#pre-train network that has ~50% accuracy
network.frame.load_state_dict(torch.load(os.path.join('E:\\', 'Abolfazl' , '2ndYearproject','code','savedNetworks','orientationClassifier')))

newnet=torch.nn.Sequential(*(list(network.frame.children()))[:-1])
newnet=torch.nn.Sequential(*(list(newnet.children()))[:-1])

def ESNFixedOri(leaky_rate,n_iterations):
    # Parameters
    learning_rate = 0.005
    n_hidden = 200
    batch_size = 1
    numOfClasses=22
    numOfFrames=330
    lastLayerSize=60
    train_leaky_rate=False
    
    class column(nn.Module):
        """
        cortical column model
        """
        def __init__(self,preTrainedModel,leaky_rate,spectral_radius = 0.9,n_hidden = n_hidden,numOfClasses=numOfClasses,lastLayerSize=lastLayerSize):
            super(column,self).__init__()
            self.frontEnd=preTrainedModel
            self.echo=etnn.LiESNCell(leaky_rate,train_leaky_rate,lastLayerSize, n_hidden, spectral_radius=0.5,nonlin_func=torch.nn.functional.relu,seed=123456)
            self.outLinear=nn.Linear(n_hidden,numOfClasses,bias=True)
        def forward(self,x,y=None,batch_size=1,lastLayerSize=lastLayerSize,numOfFrames=numOfFrames): #implement the forward pass        
            hh=np.empty((batch_size,numOfFrames,lastLayerSize))
            hh[:]=np.nan
            hh=torch.FloatTensor(hh)
            for batchNum in range(batch_size):
                m=x[batchNum,:,:,:].unsqueeze(1)
                m = self.frontEnd(m)
                hh[batchNum,:,:]=m.detach()
            x=hh    
            x=  self.echo.forward(x)
            x=  self.outLinear(x)
            return x
            
     # end class definition  
    transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.Resize((28,28)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
    
    train_set = tomImageFolderFrameSeriesAllClasses(root=os.path.join('E:\\', 'Abolfazl' , '2ndYearproject' , 'datasets','fixedOrirentation' ),transform=transform)
    
    test_set = tomImageFolderFrameSeriesAllClasses(root=os.path.join('E:\\', 'Abolfazl' , '2ndYearproject' ,  'datasets','fixedOrirentation' ),transform=transform)
    classes=('Ori1','Ori2','Ori3','Ori4','Ori5','Ori6','Ori7','Ori8','Ori9','Ori10','Ori11','Ori12','Ori13','Ori14','Ori15','Ori16''Ori17','Ori18','Ori19','Ori20','Ori21','Ori22')
    #classes=('Ori1','Ori2','Ori3','Ori4','Ori5','Ori6','Ori7','Ori8','Ori9','Ori10','Ori11','Ori12','Ori13','Ori14','Ori15','Ori16')
    
    
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,         num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True,     num_workers=2)
    
    
    # Use CUDA?
    use_cuda = False
    use_cuda = torch.cuda.is_available() if use_cuda else False
    
    # Manual seed
    mdp.numx.random.seed(1)
    np.random.seed(2)
    torch.manual_seed(1)
    
    
    #cortical column


    c1=column(newnet,leaky_rate)
    
    if use_cuda:
        c1.cuda()     
     # end if       
     
     
     # Objective function
    criterion = nn.CrossEntropyLoss()
    # Stochastic Gradient Descent
    optimizer = optim.Adam(c1.parameters(),lr=learning_rate)#, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    #freezing the pretrained front end
    freeze_layer(c1.frontEnd)
    
    # For each iteration
     #Time for printing
    training_start_time = time.time()
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
    
    #    
            # Optimize
            optimizer.step()
            # Print error measures
            print(u"Train CrossEntropyLoss: {}".format(float(loss.data)))
#            print('leaky Rate', c1.echo.leaky_rate)
#            print('output weights', param_printer(c1.outLinear)   ) 
#            print(i)
            # end for
    print(u"Time For 1 Leaky Rate: {}", (time.time() - training_start_time))
    
    
        
        
        # Test reservoir
    dataiter = iter(test_loader)
    test_u, test_y = dataiter.next()
    test_u, test_y = Variable(test_u), Variable(test_y)
    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
    y_predicted = c1(test_u)
    testResutls=torch.max(y_predicted[0],dim=1)
    showMe=testResutls[1]-test_y[0]
    numberOfMistakes=len([i for i, e in enumerate(showMe) if e != 0])
    totalNumberOfOutputs=torch.numel(showMe)
    return ((totalNumberOfOutputs-numberOfMistakes)/totalNumberOfOutputs)



#save network parameters for the record
#torch.save(network.frame.state_dict(), os.path.join('E:\\', 'Abolfazl' , '2ndYearproject','code','savedNetworks','ESNForFixedOriObj'))



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
##plot output layer
#import numpy as np
#import matplotlib.pyplot as plt
#plt.figure()
#plt.pcolor(c1.outLinear.weight.detach());
#plt.colorbar()
#plt.show()
