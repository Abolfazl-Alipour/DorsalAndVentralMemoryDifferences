#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:26:54 2020

@author: aalipour
"""


import torch
import numpy as np
import scipy.stats
import pickle
import matplotlib.pyplot as plt
from loadPretrainedOriclassifierAndIncorporateitintoESNForRotatingObjMIRO import ESNRotatingObject
from loadPretrainedOriclassifierAndIncorporateitintoESNForFixedWidthMIRO import ESNFixedWidth
from tqdm import tqdm

leaky_rateVec=torch.linspace(0.0,1.0,steps=50)
nHiddSizeVec=[20,40,80,160,320,640]
numberOfRealizations=4
n_iterations=50
resultsRotObj=np.empty([numberOfRealizations,leaky_rateVec.size()[0],len(nHiddSizeVec)])
resultsRotObj[:]=np.nan

#resultsFixedOri=np.empty([numberOfRealizations,leaky_rateVec.size()[0]])
#resultsFixedOri[:]=np.nan
resultsFixedWidth=np.empty([numberOfRealizations,leaky_rateVec.size()[0],len(nHiddSizeVec)])
resultsFixedWidth[:]=np.nan

def genrtIntegerSeeds(howManyNumbs):
    # generate random integer values
    from random import seed
    from random import randint
    # seed random number generator
    seed(randint(2, 100))
    randomSeedList=[]
    # generate some integers
    for i in range(howManyNumbs):
    	randomSeedList.append(randint(10000000, 900000000))
    
    return randomSeedList

randomSeedList=genrtIntegerSeeds(numberOfRealizations)





# for kk in tqdm(range(len(nHiddSizeVec))):
#     for j in range(len(randomSeedList)):
#         i=0
#         for leaky_rate in leaky_rateVec:
#             resultsRotObj[j,i,kk]=ESNRotatingObject(leaky_rate,n_iterations,randomSeedList[j],n_hidden =nHiddSizeVec[kk])
#             i=i+1
#             torch.cuda.empty_cache()
            
            

# # Saving the results:
# with open('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/ESNObjAllLeakingRatesCOIL100wiggle5around30.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([resultsRotObj], f)
#     # f.close()
# # # Getting back the objects:
# with open('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/ESNObjAllLeakingRatesCOIL100wiggle5around30.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     resultsRotObj = pickle.load(f)





#plt.figure()
#plt.plot(resultsRotObj)
#plt.title('Network Performance- Object Classification')
#plt.xlabel('leaky rate')
#plt.ylabel('Ratio Of correct responses to Number Of Images')
#plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
#plt.show()

for kk in tqdm(range(len(nHiddSizeVec))):
    for j in range(len(randomSeedList)):
        i=0
        for leaky_rate in leaky_rateVec:
            resultsFixedWidth[j,i,kk]=ESNFixedWidth(leaky_rate,n_iterations,randomSeedList[j],n_hidden =nHiddSizeVec[kk] ,use_cuda=True)
            i=i+1
            torch.cuda.empty_cache()
            
            

# Saving the results:
with open('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/ESNWidthAllLeakingRatesMIROMIDSIZERound2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([resultsFixedWidth], f)
    # f.close()
# # Getting back the objects:
# with open('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/ESNWidthAllLeakingRatesMIROMIDSIZE.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     resultsFixedWidth = pickle.load(f)


#plt.figure()
#plt.plot(resultsFixedWidth)
#plt.title('Network Performance- Width Classification')
#plt.xlabel('leaky rate')
#plt.ylabel('Ratio Of correct responses to Number Of Images')
#plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
#plt.show()


#i=0
#for leaky_rate in leaky_rateVec:
#    resultsFixedOri[i]=ESNFixedOri(leaky_rate)
#    i=i+1
#
#plt.figure()
#plt.plot(resultsFixedOri)
#plt.title('Network Performance- Orientation Classification')
#plt.xlabel('leaky rate')
#plt.ylabel('Ratio Of correct responses to Number Of Images')
#plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
#plt.show()
        
        
        
# resultsRotObj=resultsRotObj[0]
# for netwSize in range(len(nHiddSizeVec)):         
#     objSEM=np.empty(50)
#     obj=resultsRotObj[:,:,netwSize]
#     for i in range(obj.shape[1]):
#          objSEM[i]=(scipy.stats.sem(obj[:,i]))
         
#     objMean=np.empty(50)
#     for i in range(obj.shape[1]):
#          objMean[i]=(np.mean(obj[:,i]))
    
#     plt.figure()
#     plt.plot(range(50),objMean,"k-")
#     plt.fill_between(range(50), objMean-objSEM, objMean+objSEM,alpha=0.5, facecolor='b')
#     plt.title('Network Performance- Object Classification ({} realizations) Network Size: {}' .format(numberOfRealizations, nHiddSizeVec[netwSize]),fontsize=24)
#     plt.xlabel('leaky rate',fontsize=32)
#     plt.ylabel('Accuracy',fontsize=24)
#     plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
#     axes = plt.gca()
#     axes.set_ylim(0, 1)
#     plt.show()



# resultsFixedWidth=resultsFixedWidth[0]
# for netwSize in range(len(nHiddSizeVec)):      
#     widthSEM=np.empty(50)
#     width=resultsFixedWidth[:,:,netwSize]
#     for i in range(width.shape[1]):
#          widthSEM[i]=(scipy.stats.sem(width[:,i]))
    
#     widthMean=np.empty(50)
#     for i in range(width.shape[1]):
#          widthMean[i]=(np.mean(width[:,i]))
    
#     plt.figure()
#     plt.plot(range(50),widthMean,"k-")
#     plt.fill_between(range(50), widthMean-widthSEM, widthMean+widthSEM,alpha=0.5, facecolor='lightcoral')
#     plt.title('Width Classification ({} realizations) Network Size: {}' .format(numberOfRealizations, nHiddSizeVec[netwSize]),fontsize=24)
#     plt.xlabel('leaky rate',fontsize=32)
#     plt.ylabel('Accuracy',fontsize=24)
#     plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
#     axes = plt.gca()
# #    axes.set_ylim(0, 1)
#     plt.show()








# #plotting performance on top of each other
# for netwSize in range(len(nHiddSizeVec)):         
#     objSEM=np.empty(50)
#     obj=resultsRotObj[:,:,netwSize]
#     for i in range(obj.shape[1]):
#          objSEM[i]=(scipy.stats.sem(obj[:,i]))
         
#     objMean=np.empty(50)
#     for i in range(obj.shape[1]):
#          objMean[i]=(np.mean(obj[:,i]))
#     widthSEM=np.empty(50)
#     width=resultsFixedWidth[:,:,netwSize]
#     for i in range(width.shape[1]):
#          widthSEM[i]=(scipy.stats.sem(width[:,i]))
    
#     widthMean=np.empty(50)
#     for i in range(width.shape[1]):
#          widthMean[i]=(np.mean(width[:,i]))
    
#     plt.figure()
#     plt.plot(range(50),np.flip(objMean*100),"b-")
#     plt.fill_between(range(50), np.flip(objMean-objSEM)*100, np.flip(objMean+objSEM)*100,alpha=0.5, facecolor='b',label='Object')
#     plt.plot(range(50),np.flip(widthMean)*100,"r-")
#     plt.fill_between(range(50), np.flip(widthMean-widthSEM)*100, np.flip(widthMean+widthSEM)*100,alpha=0.5, facecolor='lightcoral',label='Orientation')

#     plt.title('Network Performance- ({} realizations) Network Size: {}' .format(numberOfRealizations, nHiddSizeVec[netwSize]),fontsize=24)
#     plt.xlabel('Memory Coeficient',fontsize=32)
#     plt.ylabel('Accuracy (%)',fontsize=24)
#     plt.xticks(ticks=[0,10, 20, 30, 40, 50 ],labels=[0,0.20, 0.40, 0.60, 0.80, 1.0 ],fontsize=22)
#     plt.yticks(ticks=[ 30,40, 50,60,70,80,90,100 ],labels=[30,40, 50,60,70,80,90,100],fontsize=22)
#     # plt.xticks(ticks=([10, 20, 30, 40, 50 ])*100,labels=([0.20, 0.40, 0.60, 0.80, 1.0 ])*100)
#     axes = plt.gca()
#     axes.set_ylim(25, 100)
#     font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 20}
#     plt.legend(loc='upper left')

#     plt.rc('font', **font)
#     plt.show()

# #end of overlaying plots



# #both plots
# plt.figure()
# plt.plot(range(50),widthMean,"k-")
# plt.fill_between(range(50), widthMean-widthSEM, widthMean+widthSEM,alpha=0.5, facecolor='lightcoral')
# plt.xlabel('leaky rate')
# plt.ylabel('Ratio Of correct responses to Number Of Images')
# plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
# plt.plot(range(50),objMean,"k-")
# plt.fill_between(range(50), objMean-objSEM, objMean+objSEM,alpha=0.5, facecolor='b')
# plt.title('Network Performance- Object Classification VS. Width Classification  (100 realizations)')
# plt.xlabel('leaky rate')
# plt.ylabel('Ratio Of correct responses to Number Of Images')
# plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 20}

# plt.rc('font', **font)
# plt.show()
