#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:54:05 2020

@author: aalipour
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:30:10 2020

@author: aalipour
"""


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dset
import torch
import mdp
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
from tomDatasetFrameSeriesSingleClassCoil100fixedWidth import tomImageFolderFrameSeriesAllClasses

numOfClasses=8

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
                     nn.Linear(in_features=60, out_features=numOfClasses),
                     nn.ReLU())
 
    

network=Network()
#pre-train network that has ~50% accuracy
network.frame.load_state_dict(torch.load('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/savedNetworks/oriClassifierMIRO_51PRCNTTrain60Test40FiftyEpochs8ClassFineGrain'))

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


class MYLSTMCellWithGateoutput(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    Special args:
    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_method='pytorch'):
        super(MYLSTMCellWithGateoutput, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        
    def forward(self, x, hidden,use_cuda=False):
        # pdb.set_trace()
        device = torch.device("cuda" if torch.cuda.is_available() &use_cuda else "cpu")
        hid, cell = hidden[1]
        hid = hid.view(hid.size(1), -1)
        cell = cell.view(cell.size(1), -1)
#        x = x.view(x.size(1), -1)
        # Linear mappings
        preact = self.i2h(x.to(device)) + self.h2h(hid.to(device))

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations

        c_t = torch.mul(cell.to(device), f_t) + torch.mul(i_t, g_t)

        h_t = torch.mul(o_t, c_t.tanh())


        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        
        #end for
        return h_t, (h_t,c_t), f_t,i_t
    
#model=MYLSTMCell(2,3)
#model(torch.rand(1,2),[torch.randn(1,3),torch.randn(1,3)])    
    
class MYLSTMWithGateoutput(nn.Module):
    
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.lstm_cell =  MYLSTMCellWithGateoutput(input_size, hidden_size,bias=True)

    def forward(self, input_, hidden=None,use_cuda=False):
        if hidden is None:
            hidden = self._init_hidden(self.lstm_cell.hidden_size)
        # input_ is of dimensionalty (n_batches, time_length, input_size, ...)
        time_length=input_.size(1)
        n_batches=input_.size(0)
        outputs = Variable(torch.zeros(n_batches, time_length,self.lstm_cell.hidden_size))
        forgetGateOutputs=torch.zeros(n_batches, time_length,self.lstm_cell.hidden_size)
        inputGateOutputs=torch.zeros(n_batches, time_length,self.lstm_cell.hidden_size)
        for batch_num in range(n_batches):
            timeStep=0
            for x in torch.unbind(input_[batch_num,:,:], dim=0):
#                pdb.set_trace()
                hidden = self.lstm_cell(x, hidden,use_cuda=use_cuda)
                outputs[batch_num,timeStep,:]=hidden[0].clone()
                forgetGateOutputs[batch_num,timeStep,:]=hidden[2]
                inputGateOutputs[batch_num,timeStep,:]=hidden[3]
                timeStep=timeStep+1
    
        return outputs,forgetGateOutputs,inputGateOutputs
    @staticmethod
    def _init_hidden(hidden_size):
        hid = torch.randn(hidden_size,1)
        cell = torch.randn(hidden_size,1)
        return hid, (hid, cell)
    
#    model=MYLSTMWithGateoutput(2,3)
#    a=model(torch.rand(4,10,2))

#loops through and calculates
def LSTMforFixedWidthForget2InputRatio(n_hidden = 200,n_epochs = 10,use_cuda = False,numOfClasses=8):
    # Parameters
    learning_rate = 0.001
    batch_size = 1
    numOfFrames=755# 378 for contineous presentation 
    lastLayerSize=60
    
    transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.Resize((28,28)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
    
    train_set = tomImageFolderFrameSeriesAllClasses(root=os.path.join('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/TrainTest/revierwersAsked/MIROMidsizeFinegrainOri8Class/train' ),transform=transform)
    
    test_set = tomImageFolderFrameSeriesAllClasses(root=os.path.join('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/TrainTest/revierwersAsked/MIROMidsizeFinegrainOri8Class/validation' ),transform=transform)
    #classes=('obj1','obj2','obj3','obj4','obj5','obj6','obj7','obj8','obj9','obj10','obj11','obj12','obj13','obj14','obj15','obj16')
    #classes=('Ori1','Ori2','Ori3','Ori4','Ori5','Ori6','Ori7','Ori8','Ori9','Ori10','Ori11','Ori12','Ori13','Ori14','Ori15','Ori16')
    
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,         num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=True,     num_workers=2)
    
    
    # Use CUDA?

    device = torch.device("cuda" if torch.cuda.is_available() &use_cuda else "cpu")
    
    
    
    #cortical column
    
    class column(nn.Module):
        """
        cortical column model
        """
        def __init__(self,preTrainedModel):
            super(column,self).__init__()
            self.frontEnd=preTrainedModel
            self.lstm=MYLSTMWithGateoutput(lastLayerSize,n_hidden)
    #        self.out=etnn.RRCell(n_hidden,16,softmax_output=False,ridge_param=0.05)
            self.outLinear=nn.Linear(n_hidden,numOfClasses,bias=True)
        def forward(self,x,y=None,use_cuda=False): #implement the forward pass 
            # device = torch.device("cuda" if torch.cuda.is_available() &use_cuda else "cpu")   
            with torch.no_grad():
                hh=np.empty((batch_size,x.size(1),lastLayerSize))
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
    #        pdb.set_trace()
            [x,forgetGate,inputGate]=  self.lstm(x,use_cuda=use_cuda)
            x=  self.outLinear(x.to(device))
    
            return x,forgetGate,inputGate
            
     # end class definition  
      
    
     
    c1=column(newnet)
    c1.to(device)
    if use_cuda:
        c1.cuda()
        c1.lstm.cuda()
        c1.lstm.lstm_cell.cuda()
     # end if       
     
     
     # Objective function
    criterion = nn.CrossEntropyLoss()               
    # Optimizer Adam for now
    optimizer = optim.Adam(c1.parameters(),lr=learning_rate)#, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    #freezing the pretrained front end
    freeze_layer(c1.frontEnd)
    
    #network.frame.load_state_dict(torch.load(os.path.join('E:\\', 'Abolfazl' , '2ndYearproject','code','savedNetworks','LSTMForRotatingObj')))
    
    # For each iteration
    for epoch in range(n_epochs):
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
    
    #        pdb.set_trace()
            out = c1(inputs,use_cuda=use_cuda)[0]
            
            loss = criterion(out.permute(0,2,1), targets.long())
    
            loss.backward(retain_graph=False)
    
    #    
            # Optimize
            optimizer.step()
            # Print error measures
#            print(u"Train CrossEntropyLoss: {}".format(float(loss.data)))
#            i=i+1
            
    #testing the performance
    correct = 0
    total = 0
    # cumFGateVals=np.empty([len(test_loader),numOfFrames,n_hidden])
    # cumIGateVals=np.empty([len(test_loader),numOfFrames,n_hidden])
    cumFGateVals=[]
    cumIGateVals=[]
    kk=0 #how many objects, used for indexing cumalative_F2IRatio 
    for images, labels in test_loader:
    #    pdb.set_trace()
        with torch.no_grad(): #this saves some GPU memory
            images, labels = Variable(images), Variable(labels)
            if use_cuda:
                outputs = c1(images.cuda(),use_cuda=True)
                _,predicted=torch.max(outputs[0][0],dim=1)
    #                showMe=predicted[1]-labels.cuda()[0]
    #                numberOfMistakes+=len([i for i, e in enumerate(showMe) if e != 0])
                cumFGateVals.append(outputs[1].squeeze())#/torch.numel(outputs[1])
                cumIGateVals.append(outputs[2].squeeze())# I don't know why it was np.max(forgetGate2inputGateRatio.data.numpy()) itshould be np.mean (), so I changed it
                 
                kk+=1            
                correct += (predicted == labels.cuda()).sum()
                total += labels.size(1)
                del labels,outputs
                torch.cuda.empty_cache()
                

            else:
                 outputs = c1(images)
                 _,predicted=torch.max(outputs[0][0],dim=1)
    #             print(torch.min(outputs[1][0],dim=1))
                 correct += (predicted == labels).sum()
                 total += labels.size(1)
    
                 cumFGateVals.append(outputs[1].squeeze())#/torch.numel(outputs[1])
                 cumIGateVals.append(outputs[2].squeeze())# I don't know why it was np.max(forgetGate2inputGateRatio.data.numpy()) itshould be np.mean (), so I changed it
                 
                 kk+=1
    correct=correct.cpu().numpy()
    total=(torch.tensor(total)).numpy()
    accVec=(correct/total)
    return(accVec,cumFGateVals,cumIGateVals)
    
nHiddSizeVec=[20,40,80,160,320,640]#[1280,2560]#
numberOfRealizations=30
n_epochs = 200
numOfWidthClasses=4
numOfFrames=755
accResultLSTMFixedWidth=np.empty([len(nHiddSizeVec),numberOfRealizations])
accResultLSTMFixedWidth[:]=np.nan

cumFGateValsMIROOri=[]
cumIGateValsMIROOri=[]

for ii in tqdm(range(len(nHiddSizeVec))):
    print(ii)
    cumFGateValsOfSize=[]
    cumIGateValsOfSize=[]
    for kk in range(numberOfRealizations):
        accResultLSTMFixedWidth[ii,kk],tmpCumFGateVals,tmpCumIGateVals=LSTMforFixedWidthForget2InputRatio(n_hidden = nHiddSizeVec[ii],n_epochs=n_epochs,use_cuda = True,numOfClasses=8)
        cumFGateValsOfSize.append(tmpCumFGateVals)
        cumIGateValsOfSize.append(tmpCumIGateVals)
    cumFGateValsMIROOri.append(cumFGateValsOfSize)
    cumIGateValsMIROOri.append(cumIGateValsOfSize)







import pickle


# Saving the results:
with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/MIROOriTrainTest200Epoch10-3LerRateMidSizeFG.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([accResultLSTMFixedWidth, cumFGateValsMIROOri, cumIGateValsMIROOri], f)
    f.close()
# Getting back the objects:
with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/MIROOriTrainTest200Epoch10-3LerRateMidSizeFG.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMFixedWidth, cumFGateValsMIROOri, cumIGateValsMIROOri = pickle.load(f)


