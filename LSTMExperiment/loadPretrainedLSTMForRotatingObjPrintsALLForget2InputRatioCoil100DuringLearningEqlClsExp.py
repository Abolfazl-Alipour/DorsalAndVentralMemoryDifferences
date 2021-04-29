# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:38:40 2020

@author: aalipour
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:37:50 2019

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
import mdp
import pdb
from tqdm import tqdm
from tomDatasetFrameSeriesAllClassesObjIdentityCoil100 import tomImageFolderFrameSeriesAllClasses
numOfClasses=54
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
# network.frame.load_state_dict(torch.load(os.path.join('geode2','home','u010','aalipour','Carbonate','Downloads','2ndYearproject','2020Wave','code','savedNetworks','TestTrain','objectClassifierCoil100_200epochs53PRCNTTestTrain')))
network.frame.load_state_dict(torch.load('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/savedNetworks/objClassifierCOIL100_72PRCNTTrain60Test40FiftyEpochs30DegRevAskEqlClsExp'))
newnet=torch.nn.Sequential(*(list(network.frame.children()))[:-1])
newnet=torch.nn.Sequential(*(list(newnet.children()))[:-1])

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
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        
    def forward(self, x, hidden,use_cuda=False):
        
        device = torch.device("cuda" if torch.cuda.is_available() &use_cuda else "cpu")
        hid, cell = hidden[1]
        hid = hid.view(hid.size(1), -1)
        cell = cell.view(cell.size(1), -1)
        hid.to(device)
        cell.to(device)
        # # self.input_size.to(device)
        # # self.hidden_size.to(device)
        # # self.bias.to(device)
        # # self.dropout.to(device)
        # self.i2h.to(device)
        # self.h2h.to(device)
          
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
        # pdb.set_trace()
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
        # self.lstm_cell.to(device)
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
                # pdb.set_trace()
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
    
#model=MYLSTMWithGateoutput(2,3)
#a=model(torch.rand(4,10,2))
def LSTMforRotObjForget2InputRatio(n_hidden = 200,n_epochs = 1,use_cuda = False,numOfClasses=54):
# Parameters

    learning_rate = 0.0001 #the learning reate had been lowered in here to improve accuracy
    batch_size = 1
    
    # numOfFrames=72
    lastLayerSize=60
    
    
    
    transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.Resize((28,28)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
    
    train_set = tomImageFolderFrameSeriesAllClasses('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/TrainTest/revierwersAsked/Coil100Obj30DegWigglePlusMinus5SixtyFortyRawEqlClsExp/train',transform=transform)
    
    test_set = tomImageFolderFrameSeriesAllClasses('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/TrainTest/revierwersAsked/Coil100Obj30DegWigglePlusMinus5SixtyFortyRawEqlClsExp/validation' ,transform=transform)
#    classes=('obj1','obj2','obj3','obj4','obj5','obj6','obj7','obj8','obj9','obj10','obj11','obj12','obj13','obj14','obj15','obj16')
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
            device = torch.device("cuda" if torch.cuda.is_available() &use_cuda else "cpu")       
            hh=np.empty((batch_size,x.size(1),lastLayerSize))
            hh[:]=np.nan
            hh=torch.FloatTensor(hh)
            # pdb.set_trace()
            for batchNum in range(batch_size):
                m=x[batchNum,:,:,:].unsqueeze(1)
                m = self.frontEnd(m)
                hh[batchNum,:,:]=m.detach()
            x=hh    
            # pdb.set_trace()
            
            [x,forgetGate,inputGate]=  self.lstm(x,use_cuda=use_cuda)
            x=  self.outLinear(x.to(device))
    
            return x,forgetGate,inputGate
            
     # end class definition  
      
    
     
    c1=column(newnet)
    c1.to(device)
    
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
#        print('here{}'.format(float(n_iterations)))
        # Iterate over batches
#        i=1
        for data in train_loader:
            # Inputs and outputs
            inputs, targets = data
            inputs, targets = Variable(inputs), Variable(targets)
            inputs, targets = inputs.to(device), targets.to(device)
        
            # Gradients to zero
            optimizer.zero_grad()
        
            # Forward
    
            # pdb.set_trace()
            out = c1(inputs,use_cuda=use_cuda)[0]
            
            loss = criterion(out.permute(0,2,1), targets.long())
    
            loss.backward(retain_graph=False)
    
    #    
            # Optimize
            optimizer.step()
            # Print error measures
#            print(u"Train CrossEntropyLoss: {}".format(float(loss.data)))
#            i=i+1
#            print(i)
    
    correct = 0
    total = 0
    # cumFGateVals=np.empty([len(test_loader),numOfFrames,n_hidden])
    # cumIGateVals=np.empty([len(test_loader),numOfFrames,n_hidden])
    cumFGateVals=[]
    cumIGateVals=[]
    kk=0 #how many objects, used for indexing cumalative_F2IRatio 
    for images, labels in test_loader:
        # pdb.set_trace()
        with torch.no_grad(): #this saves some GPU memory
            images, labels = Variable(images), Variable(labels)
            if use_cuda:
                outputs = c1(images.cuda(),use_cuda=True)
                _,predicted=torch.max(outputs[0][0],dim=1)
    #                showMe=predicted[1]-labels.cuda()[0]
    #                numberOfMistakes+=len([i for i, e in enumerate(showMe) if e != 0])
                correct += (predicted == labels.cuda()).sum()
                total += labels.size(1)
                cumFGateVals.append(outputs[1].squeeze())#/torch.numel(outputs[1])
                cumIGateVals.append(outputs[2].squeeze())# I don't know why it was np.max(forgetGate2inputGateRatio.data.numpy()) itshould be np.mean (), so I changed it
                kk+=1
                del labels,outputs
                torch.cuda.empty_cache()
            else:
                 outputs = c1(images)
                 _,predicted=torch.max(outputs[0][0],dim=1)
    
                 correct += (predicted == labels).sum()
                 total += labels.size(1)
                 cumFGateVals.append(outputs[1].squeeze())#/torch.numel(outputs[1])
                 cumIGateVals.append(outputs[2].squeeze())# I don't know why it was np.max(forgetGate2inputGateRatio.data.numpy()) itshould be np.mean (), so I changed it
                 kk+=1
#                 print(np.isinf(forgetGate2inputGateRatioMat.data.numpy()).any())
#                 print(kk)
    correct=correct.cpu().numpy()
    total=(torch.tensor(total)).numpy()
    accVec=(correct/total)
    return(accVec,cumFGateVals,cumIGateVals) 

nHiddSizeVec=[20,40,80,160,320,640]
numberOfRealizations=30
n_epochs=200#40
# numOfFrames=72
numOfObjClasses=54
accResultLSTMRotObj=np.empty([len(nHiddSizeVec),numberOfRealizations])
accResultLSTMRotObj[:]=np.nan
cumFGateValsCoilObj=[]#len(nHiddSizeVec),numberOfRealizations,numOfWidthClasses,[],nHiddSizeVec[-1]]
cumIGateValsCoilObj=[]

for ii in tqdm(range(len(nHiddSizeVec))):
    print(ii)
    cumFGateValsOfSize=[]
    cumIGateValsOfSize=[]
    for kk in range(numberOfRealizations):
        accResultLSTMRotObj[ii,kk],tmpCumFGateVals,tmpCumIGateVals=LSTMforRotObjForget2InputRatio(n_hidden = nHiddSizeVec[ii],n_epochs=n_epochs,use_cuda=True)
        cumFGateValsOfSize.append(tmpCumFGateVals)
        cumIGateValsOfSize.append(tmpCumIGateVals)
    cumFGateValsCoilObj.append(cumFGateValsOfSize)
    cumIGateValsCoilObj.append(cumIGateValsOfSize)


import pickle


# Saving the results:
with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/Coil100ObjTrainTest200Epoch10-4LerRate30deg5degwiggleEqlClsExp.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([accResultLSTMRotObj, cumFGateValsCoilObj, cumIGateValsCoilObj], f)
    f.close()
# Getting back the objects:
with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/Coil100ObjTrainTest200Epoch10-4LerRate30deg5degwiggleEqlClsExp.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMRotObj, cumFGateValsCoilObj, cumIGateValsCoilObj = pickle.load(f)





# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats
# RotObjSEM=np.empty(6)
# RotObjSEM[:]=np.nan
# RotObj=F2IRatioRotObj
# for i in range(RotObj.shape[0]):
#      RotObjSEM[i]=(scipy.stats.sem(RotObj[i,:]))

# RotObjhMean=np.empty(6)
# for i in range(RotObj.shape[0]):
#      RotObjhMean[i]=(np.mean(RotObj[i,:]))
    
# plt.figure()

# plt.errorbar(np.arange(6),RotObjhMean,yerr=RotObjSEM,fmt='s',color='b',capsize=10,ms=10)
# #plt.fill_between(range(50), widthMean-widthSTD, widthMean+widthSTD,alpha=0.5, facecolor='lightcoral')
# plt.title('Mean Forget Gate to Input Gate Ratio ({} realizations)' .format(numberOfRealizations),fontsize=24)
# plt.xlabel('Network Size',fontsize=32)
# plt.ylabel('Mean Forget Gate to Input Gate Ratio',fontsize=24)
# plt.xticks(ticks=[0,1,2,3,4,5 ],labels=[20, 40, 80, 160,320,640],fontsize=24)
# matplotlib.rc('ytick', labelsize=32) 
# #axes = plt.gca()
# #axes.set_ylim(0, 1)
# plt.yscale('log')
# plt.show()

# ##accuracy
# widthSTD=np.empty(6)
# widthSTD[:]=np.nan
# width=accResultLSTMRotObj
# for i in range(width.shape[0]):
#      widthSTD[i]=(scipy.stats.sem(width[i,:]))

# widthMean=np.empty(6)
# for i in range(width.shape[0]):
#      widthMean[i]=(np.mean(width[i,:]))
    
# plt.figure()


# #  
# ## Test reservoir
# #dataiter = iter(test_loader)
# #test_u, test_y = dataiter.next()
# #test_u, test_y = Variable(test_u), Variable(test_y)
# #if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
# #y_predicted = c1(test_u)
# ##Here you can compute the ratios between two gates
# #forgetGate2inputGateRatio=((y_predicted[1]/y_predicted[2]).squeeze())/torch.numel(y_predicted[1])
# #np.mean(forgetGate2inputGateRatio.data.numpy())# I don't know why it was np.max(forgetGate2inputGateRatio.data.numpy()) itshould be np.mean (), so I changed it
# #
# #if you like to plot the ratios and activity weights, use these lines:
# x=forgetGate2inputGateRatio.data.numpy()
# plt.scatter(range(4400),np.array(torch.Tensor(x).view(4400,-1)))
# plt.show()
# #end
# #
# #testResutls=torch.max(y_predicted[0],dim=1)
# #showMe=testResutls[1]-test_y[0]
# #[i for i, e in enumerate(showMe) if e != 0]
# #hidden2forgetGateNormalizedSum=(sum(sum(c1.lstm.weight_hh_l0[200:400])))/(torch.numel(c1.lstm.weight_hh_l0[200:400]))
# #Input2InputgateNormalizedSum=(sum(sum(c1.lstm.weight_ih_l0[0:199])))/(torch.numel(c1.lstm.weight_ih_l0[0:199]))
# #forget2inputRatio=hidden2forgetGateNormalizedSum/Input2InputgateNormalizedSum
# #
# #
# ##save network parameters for the rocord
# #torch.save(network.frame.state_dict(), os.path.join('E:\\', 'Abolfazl' , '2ndYearproject','code','savedNetworks','LSTMForRotatingObj'))
# #
# #
# ##function for ploting convolutional kernels
# #def plot_kernels(tensor, num_cols=6):
# #    num_kernels = tensor.shape[0]
# #    num_rows = num_kernels // num_cols
# #    fig = plt.figure(figsize=(num_cols,num_rows))
# #    for i in range(num_kernels):
# #        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
# #        ax1.imshow(tensor[i][0,:,:], cmap='gray')
# #        ax1.axis('off')
# #        ax1.set_xticklabels([])
# #        ax1.set_yticklabels([])
# ##ploting the first conv2d layer's kernels
# #plot_kernels(list(c1.frontEnd.state_dict().values())[0])
# #
# #
# ##plot output layer
# #plt.imshow(c1.outLinear.weight.detach())
