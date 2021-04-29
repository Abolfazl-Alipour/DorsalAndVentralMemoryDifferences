# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:42:50 2019

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

class Flatten(torch.nn.Module):
  def forward(self, x):
      batch_size = x.shape[0]
      return x.view(batch_size, -1)
  
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
network.frame.load_state_dict(torch.load(os.path.join('E:\\', 'Abolfazl' , '2ndYearproject','code','savedNetworks','objectClassifier')))

newnet=torch.nn.Sequential(*(list(network.frame.children()))[:-1])
newnet=torch.nn.Sequential(*(list(newnet.children()))[:-1])

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
leaky_rate = 0.5
learning_rate = 0.005
firstLayerSize = 5
n_hidden = 200
n_iterations = 200
train_sample_length = 5000
test_sample_length = 1000
n_train_samples = 2
n_test_samples = 1
batch_size = 4
momentum = 0.95
weight_decay = 0
numOfClasses=16
numOfFrames=336
lastLayerSize=60
train_leaky_rate=False
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
from tomDatasetFrameSeriesAllClassesObjIdentity import tomImageFolderFrameSeriesAllClasses


transform = transforms.Compose(
[transforms.Grayscale(num_output_channels=1),
 transforms.Resize((28,28)),
 transforms.ToTensor(),
 transforms.Normalize((0.5,), (0.5,))])

train_set = tomImageFolderFrameSeriesAllClasses(root=os.path.join('E:\\', 'Abolfazl' , '2ndYearproject' , 'datasets','fixedObjIdentity' ),transform=transform)

test_set = tomImageFolderFrameSeriesAllClasses(root=os.path.join('E:\\', 'Abolfazl' , '2ndYearproject' ,  'datasets','fixedObjIdentity' ),transform=transform)
classes=('obj1','obj2','obj3','obj4','obj5','obj6','obj7','obj8','obj9','obj10','obj11','obj12','obj13','obj14','obj15','obj16')
#classes=('Ori1','Ori2','Ori3','Ori4','Ori5','Ori6','Ori7','Ori8','Ori9','Ori10','Ori11','Ori12','Ori13','Ori14','Ori15','Ori16')


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,         num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=True,     num_workers=2)


# Use CUDA?
use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed
mdp.numx.random.seed(1)
np.random.seed(2)
torch.manual_seed(1)

def param_printer(layer):
 for param in layer.parameters():
  print(param) 

def printIfReqGrad(layer):
    for param in layer.parameters(): 
        print(param.requires_grad)
  
  
def freeze_layer(layer):
 for param in layer.parameters():
  param.requires_grad = False

#cortical column

class column(nn.Module):
    """
    cortical column model
    """
    def __init__(self,preTrainedModel):
        super(column,self).__init__()
        self.frontEnd=preTrainedModel
        self.lstm=torch.nn.LSTM(lastLayerSize,n_hidden,batch_first=True)
#        self.out=etnn.RRCell(n_hidden,16,softmax_output=False,ridge_param=0.05)
        self.outLinear=nn.Linear(n_hidden,numOfClasses,bias=True)
    def forward(self,x,y=None): #implement the forward pass        
        hh=np.empty((batch_size,numOfFrames,lastLayerSize))
        hh[:]=np.nan
        hh=torch.FloatTensor(hh)
#        pdb.set_trace()
        for batchNum in range(batch_size):
            m=x[batchNum,:,:,:].unsqueeze(1)
            m = self.frontEnd(m)
            hh[batchNum,:,:]=m.detach()
        x=hh    
        pdb.set_trace()
        x, (h_n,c_n)=  self.lstm(x)
        x=  self.outLinear(x)
#        if y is not None:
#            frame=torch.zeros(batch_size,numOfFrames,numOfClasses)
#            if use_cuda: frame.cuda()
#            for batchNum in range(batch_size):
#                frame[batchNum,0,y[batchNum]]=1
#            y=frame
#            tmp=self.out(x,y)
#            self.out.finalize()
#            x=self.out(x)
#            return x
#        else:
        return x #, y)
        
 # end class definition  
  

 
c1=column(newnet)

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
for epoch in range(n_iterations):
    # Iterate over batches
    i=1
    for data in train_loader:
        # Inputs and outputs
        inputs, targets = data
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
    
        # Gradients to zero
        optimizer.zero_grad()
    
        # Forward

#        pdb.set_trace()
        out = c1(inputs,targets)
#        frame=torch.zeros(batch_size,numOfFrames,numOfClasses,device='cuda')
#        for batchNum in range(batch_size):
#            frame[batchNum,0,targets[batchNum]]=1
##        
#        targets=frame
#        newout=torch.empty((batch_size,numOfClasses,numOfFrames))
#        for i in range(4):
#            out[i,:,:]=out[i,:,:].t()
#        if use_cuda: newout = newout.cuda()
        loss = criterion(out.permute(0,2,1), targets.long())


        loss.backward(retain_graph=True)

#    
        # Optimize
        optimizer.step()
        # Print error measures
        print(u"Train CrossEntropyLoss: {}".format(float(loss.data)))
        i=i+1
        print('Forget Gate Sum', sum(sum(c1.lstm.weight_hh_l0[200:400])))
#        print('output weights', param_printer(c1.outLinear)   )   
#        if i>20:
#            break
        
#        c1.out.training=True
        # end for


    
    
    # Test reservoir
dataiter = iter(test_loader)
test_u, test_y = dataiter.next()
test_u, test_y = Variable(test_u), Variable(test_y)
if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
y_predicted = c1(test_u)
testResutls=torch.max(y_predicted[0],dim=1)
showMe=testResutls[1]-test_y[0]
[i for i, e in enumerate(showMe) if e != 0]
forgetHiddenGateNormalizedSum=(sum(sum(c1.lstm.weight_hh_l0[200:400])))/(torch.numel(c1.lstm.weight_hh_l0[200:400]))
forgetInputGateNormalizedSum=(sum(sum(c1.lstm.weight_ih_l0[200:400])))/(torch.numel(c1.lstm.weight_ih_l0[200:400]))
forget2inputRatio=forgetHiddenGateNormalizedSum/forgetInputGateNormalizedSum

#save network parameters for the rocord
torch.save(network.frame.state_dict(), os.path.join('E:\\', 'Abolfazl' , '2ndYearproject','code','savedNetworks','LSTMForRotatingObj'))


#function for ploting convolutional kernels
def plot_kernels(tensor, num_cols=6):
    num_kernels = tensor.shape[0]
    num_rows = num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i][0,:,:], cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
#ploting the first conv2d layer's kernels
plot_kernels(list(c1.frontEnd.state_dict().values())[0])


#plot output layer
plt.imshow(c1.outLinear.weight.detach())
