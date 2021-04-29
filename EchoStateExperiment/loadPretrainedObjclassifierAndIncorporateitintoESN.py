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
leaky_rate = 1.0
learning_rate = 10.001
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
numOfFrames=1
numOfClasses=16
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


transform = transforms.Compose(
[transforms.Grayscale(num_output_channels=1),
 transforms.Resize((28,28)),
 transforms.ToTensor(),
 transforms.Normalize((0.5,), (0.5,))])

train_set = dset.ImageFolder(root=os.path.join('E:\\', 'Abolfazl' , '2ndYearproject' , 'ImagesGeneratedTom' , 'rotatedStims' ),transform=transform)

test_set = dset.ImageFolder(root=os.path.join('E:\\','Abolfazl','2ndYearproject','ImagesGeneratedTom','rotatedStims'),transform=transform)

classes=('Ori1','Ori2','Ori3','Ori4','Ori5','Ori6','Ori7','Ori8','Ori9','Ori10','Ori11','Ori12','Ori13','Ori14','Ori15','Ori16')


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,         num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=True,     num_workers=2)


# Use CUDA?
use_cuda = True
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed
mdp.numx.random.seed(1)
np.random.seed(2)
torch.manual_seed(1)

def param_printer(layer):
 for param in layer.parameters():
  print(param) 

#cortical column

class column(nn.Module):
    """
    cortical column model
    """
    def __init__(self,preTrainedModel):
        super(column,self).__init__()
        self.frontEnd=preTrainedModel
        self.echo=etnn.LiESNCell(1,False,60, n_hidden, spectral_radius=0.9,seed=123456789)
        self.out=etnn.RRCell(n_hidden,16,softmax_output=True,ridge_param=0.05)
    def forward(self,x,y=None): #implement the forward pass        
        x = self.frontEnd(x)
        x=  self.echo.forward(x.unsqueeze(1))
        if y is not None:
            frame=torch.zeros(batch_size,numOfFrames,numOfClasses,device='cuda')
            for batchNum in range(batch_size):
                frame[batchNum,0,y[batchNum]]=1
            y=frame
            tmp=self.out(x,y)
            self.out.finalize()
            x=self.out(x)
            return x
        else:
            return self.out(x, y)
        
 # end class definition  

 
c1=column(newnet)

if use_cuda:
    c1.cuda()     
 # end if       
 
 
 # Objective function
criterion = nn.CrossEntropyLoss()
# Stochastic Gradient Descent
optimizer = optim.Adam(c1.parameters(),lr=learning_rate)#, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


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
        loss = criterion(out.squeeze(1), targets.long())


        loss.backward(retain_graph=True)

#    
        # Optimize
        optimizer.step()
        # Print error measures
        print(u"Train CrossEntropyLoss: {}".format(float(loss.data)))
        i=i+1
        c1.out.reset()
        # end for
    

    
    
    # Test reservoir
    dataiter = iter(test_loader)
    test_u, test_y = dataiter.next()
    test_u, test_y = Variable(test_u), Variable(test_y)
    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
    y_predicted = c1(test_u)

    # Print error measures
    print(u"Test MSE: {}".format(echotorch.utils.mse(y_predicted.data, test_y.data)))
    print(u"Test NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, test_y.data)))
    print(u"")
    # end for