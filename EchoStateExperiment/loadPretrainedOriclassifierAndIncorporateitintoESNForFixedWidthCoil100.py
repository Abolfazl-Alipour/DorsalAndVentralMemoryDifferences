# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:17:02 2019

@author: aalipour
"""


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import torch.optim as optim
import time
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dset
import torch
import echotorch.nn as etnn
import echotorch.utils
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import mdp
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
from tomDatasetFrameSeriesSingleClassCoil100fixedWidth import tomImageFolderFrameSeriesAllClasses



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
network.frame.load_state_dict(torch.load(os.path.join('E:\\', 'Abolfazl' , '2ndYearproject','code','savedNetworks','orientationClassifier')))

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
  
#  # Manual seed
#mdp.numx.random.seed(1)
#np.random.seed(2)
#torch.manual_seed(1)
#    


  




def ESNFixedWidth(leaky_rate=0.5,n_iterations=140,seed=72347,n_hidden = 200,use_cuda=True):
    # Parameters
    spectral_radius = 0.9
    learning_rate = 0.005
    batch_size = 1
    numOfClasses=6
    numOfFrames=755
    lastLayerSize=60
    train_leaky_rate=True
    class column(nn.Module):
        """
        cortical column model
        """
        def __init__(self,preTrainedModel,leaky_rate,spectral_radius = spectral_radius,n_hidden = n_hidden,numOfClasses=numOfClasses,lastLayerSize=lastLayerSize):
            super(column,self).__init__()
            self.frontEnd=preTrainedModel
            self.echo=etnn.LiESNCell(leaky_rate,train_leaky_rate,lastLayerSize, n_hidden, spectral_radius=0.5,nonlin_func=torch.nn.functional.relu,seed=seed)
            self.outLinear=nn.Linear(n_hidden,numOfClasses,bias=True)
        def forward(self,x,y=None,batch_size=1,lastLayerSize=60,numOfFrames=numOfFrames): #implement the forward pass        
            with torch.no_grad():
                hh=np.empty((batch_size,numOfFrames,lastLayerSize))
                hh[:]=np.nan
                hh=torch.FloatTensor(hh)
        #        pdb.set_trace()
                for batchNum in range(batch_size):
                    m=x[batchNum,:,:,:].unsqueeze(1)
                    m = self.frontEnd(m)
                    hh[batchNum,:,:]=m.detach()
                if use_cuda:
                    x=hh.cuda() 
                else:
                    x=hh
                x=x.detach()
            x=  self.echo.forward(x)
            x=  self.outLinear(x)
            return x
    
    transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.Resize((28,28)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
    
    train_set = tomImageFolderFrameSeriesAllClasses(root=os.path.join('E:\\', 'Abolfazl' , '2ndYearproject' , 'datasets','coil-100Width' ),transform=transform)
    
    test_set = tomImageFolderFrameSeriesAllClasses(root=os.path.join('E:\\', 'Abolfazl' , '2ndYearproject' , 'datasets','coil-100Width'),transform=transform)
    classes=('Ori1','Ori2','Ori3','Ori4','Ori5','Ori6')
    #classes=('Ori1','Ori2','Ori3','Ori4','Ori5','Ori6','Ori7','Ori8','Ori9','Ori10','Ori11','Ori12','Ori13','Ori14','Ori15','Ori16')
    
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,         num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=True,     num_workers=2)
    
    
    # Use CUDA?
    use_cuda = torch.cuda.is_available() if use_cuda else False
    

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
#    training_start_time=time.time()
    for epoch in range(n_iterations):
        # Iterate over batches
#        i=1
        for data in train_loader:
            # Inputs and outputs
            inputs, targets = data
            inputs, targets = Variable(inputs), Variable(targets)
            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        
            # Gradients to zero
            optimizer.zero_grad()
            # Forward
    
            out = c1(inputs,targets)
#            pdb.set_trace()
            loss = criterion(out.permute(0,2,1), targets.long())
    
    
            loss.backward(retain_graph=False)
    
    #    
            # Optimize
            optimizer.step()
            del inputs,targets,out,loss
            torch.cuda.empty_cache()
            # Print error measures
#            print(u"Train CrossEntropyLoss: {}".format(float(loss.data)))
#            i+=1
#            print('leaky Rate', c1.echo.leaky_rate)
#            print('output weights', param_printer(c1.outLinear)   ) 
#            print(i)
            # end for
#    print(u"Time For 1 Leaky Rate: ", (time.time() - training_start_time))
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
    return (correct/total)#accur 
#        # Test reservoir
#    dataiter = iter(test_loader)
#    test_u, test_y = dataiter.next()
#    test_u, test_y = Variable(test_u), Variable(test_y)
#    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
#    y_predicted = c1(test_u)
#    testResutls=torch.max(y_predicted[0],dim=1)
#    showMe=testResutls[1]-test_y[0]
#    numberOfMistakes=len([i for i, e in enumerate(showMe) if e != 0])
#    totalNumberOfOutputs=torch.numel(showMe)
#    return ((totalNumberOfOutputs-numberOfMistakes)/totalNumberOfOutputs)

##save network parameters for the record
#torch.save(network.frame.state_dict(), os.path.join('E:\\', 'Abolfazl' , '2ndYearproject','code','savedNetworks','ESNForFixedOriObj'))
#
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
##plot output layer
#import numpy as np
#import matplotlib.pyplot as plt
#plt.figure()
#plt.pcolor(c1.outLinear.weight.detach());
#plt.colorbar()
#plt.show()
