# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:52:58 2019

@author: aalipour
"""


# Author: Robert Guthrie

import math
import torch as th
import torch.nn as nn

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

'''
STEP 1: LOADING DATASET
'''
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

'''
STEP 3: CREATE MODEL CLASS
'''

#class LSTMModel(nn.Module):
#    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#        super(LSTMModel, self).__init__()
#        # Hidden dimensions
#        self.hidden_dim = hidden_dim
#
#        # Number of hidden layers
#        self.layer_dim = layer_dim
#
#        # Building your LSTM
#        # batch_first=True causes input/output tensors to be of shape
#        # (batch_dim, seq_dim, feature_dim)
#        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
#
#        # Readout layer
#        self.fc = nn.Linear(hidden_dim, output_dim)
#
#    def forward(self, x):
#        # Initialize hidden state with zeros
#        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
#
#        # Initialize cell state
#        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
#
#        # One time step
#        # We need to detach as we are doing truncated backpropagation through time (BPTT)
#        # If we don't, we'll backprop all the way to the start even after going through another batch
#        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
#
#        # Index hidden state of last time step
#        # out.size() --> 100, 28, 100
#        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
#        out = self.fc(out[:, -1, :]) 
#        # out.size() --> 100, 10
#        return out

import math
import torch as th
import torch.nn as nn



class MYLSTM(nn.Module):

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
        super(MYLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input_, hidden=None):
        if hidden is None:
            hidden = self._init_hidden(self.hidden_size)
        do_dropout = self.training and self.dropout > 0.0
        time_length=input_.size(1)
        n_batches=input_.size(0)
        outputs = Variable(torch.zeros(n_batches, time_length,self.hidden_size))
#        pdb.set_trace()
        for batchNumb in range(n_batches):
            h, c = hidden
            h = h.view(h.size(1), -1)
            c = c.view(c.size(1), -1)
            x = input_[batchNumb,:,:]#.view(x[batchNumb,:,:].size(1), -1)
    
            # Linear mappings
            preact = self.i2h(x) + self.h2h(h)
    
            # activations
            gates = preact[:, :3 * self.hidden_size].sigmoid()
            g_t = preact[:, 3 * self.hidden_size:].tanh()
            i_t = gates[:, :self.hidden_size]
            f_t = gates[:, self.hidden_size:2 * self.hidden_size]
            o_t = gates[:, -self.hidden_size:]
    
            # cell computations
            if do_dropout and self.dropout_method == 'semeniuta':
                g_t = F.dropout(g_t, p=self.dropout, training=self.training)
    
            c_t = th.mul(c, f_t) + th.mul(i_t, g_t)
    
            if do_dropout and self.dropout_method == 'moon':
                    c_t.data.set_(th.mul(c_t, self.mask).data)
                    c_t.data *= 1.0/(1.0 - self.dropout)
    
            h_t = th.mul(o_t, c_t.tanh())
    
            # Reshape for compatibility
            if do_dropout:
                if self.dropout_method == 'pytorch':
                    F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
                if self.dropout_method == 'gal':
                        h_t.data.set_(th.mul(h_t, self.mask).data)
                        h_t.data *= 1.0/(1.0 - self.dropout)
    
            h_t = h_t.view(1, h_t.size(0), -1)
            c_t = c_t.view(1, c_t.size(0), -1)
            
            outputs[batchNumb,:,:]=h_t
            #end for
        return outputs
    #have to work on it
    @staticmethod
    def _init_hidden(hidden_size):
        hid = th.randn(hidden_size,1)
        cell = th.randn(hidden_size,1)
        return hid, cell
    
model = MYLSTM(3, 4)   
a=model(torch.randn(1,25,3))
[torch.randn(4,1),torch.randn(4,1)])


'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 28
hidden_dim = 100
layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 10

model = MYLSTM(input_dim, hidden_dim)

# JUST PRINTING MODEL & PARAMETERS 
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1

optimizer = th.optim.SGD(model.parameters(), lr=learning_rate)  

'''
STEP 7: TRAIN THE MODEL
'''

# Number of steps to unroll
seq_dim = 28  

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as torch tensor with gradient accumulation abilities
        images = images.view(-1, seq_dim, input_dim).requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        pdb.set_trace()
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Resize image
                images = images.view(-1, seq_dim, input_dim)

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))



























































