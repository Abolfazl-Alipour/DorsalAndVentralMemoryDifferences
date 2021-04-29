# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:27:31 2019

@author: aalipour
"""
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from loadPretrainedOriclassifierAndIncorporateitintoESNForRotatingObj import ESNRotatingObject
#from loadPretrainedOriclassifierAndIncorporateitintoESNForFixedOriObj import *
from loadPretrainedOriclassifierAndIncorporateitintoESNForFixedWidth import ESNFixedWidth
from tqdm import tqdm

forgetRateVec=torch.linspace(0.0,1.0,steps=50)
nHiddSizeVec=[20,40,80,160,320,640]#[20]#
numberOfRealizations=15
n_iterations=30
resultsRotObj=np.empty([numberOfRealizations,forgetRateVec.size()[0],len(nHiddSizeVec)])
resultsRotObj[:]=np.nan

#resultsFixedOri=np.empty([numberOfRealizations,leaky_rateVec.size()[0]])
#resultsFixedOri[:]=np.nan
resultsFixedWidth=np.empty([numberOfRealizations,forgetRateVec.size()[0],len(nHiddSizeVec)])
resultsFixedWidth[:]=np.nan

def genrtIntegerSeeds(howManyNumbs):
    # generate random integer values
    from random import seed
    from random import randint
    # seed random number generator
    seed(1)
    randomSeedList=[]
    # generate some integers
    for i in range(howManyNumbs):
    	randomSeedList.append(randint(10000000, 900000000))
    
    return randomSeedList

randomSeedList=genrtIntegerSeeds(numberOfRealizations)
for kk in tqdm(range(len(nHiddSizeVec))):
    for j in range(len(randomSeedList)):
        i=0
        for forgetRate in forgetRateVec:
            resultsRotObj[j,i,kk]=ESNRotatingObject(forgetRate,n_iterations,randomSeedList[j],n_hidden =nHiddSizeVec[kk] )
            i=i+1
            torch.cuda.empty_cache()
            
            

# Saving the results:
with open('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ESNObjAllLeakingRates.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([resultsRotObj], f)
    # f.close()
# # Getting back the objects:
# with open('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ESNObjAllLeakingRates.pkl','rb') as f:  # Python 3: open(..., 'rb')
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
        for forgetRate in forgetRateVec:
            resultsFixedWidth[j,i,kk]=ESNFixedWidth(forgetRate,n_iterations,randomSeedList[j],n_hidden =nHiddSizeVec[kk] )
            i=i+1
            torch.cuda.empty_cache()


# Saving the results:
with open('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ESNWidthAllLeakingRates.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([resultsFixedWidth], f)
    # f.close()
# # Getting back the objects:
# with open('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ESNWidthAllLeakingRates.pkl','rb') as f:  # Python 3: open(..., 'rb')
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
        
        
  ############################################################################################      
        
# for netwSize in range(len(nHiddSizeVec)):         
#     objSTD=np.empty(50)
#     obj=resultsRotObj[:,:,netwSize]
#     for i in range(obj.shape[1]):
#          objSTD[i]=(np.std(obj[:,i]))
         
#     objMean=np.empty(50)
#     for i in range(obj.shape[1]):
#          objMean[i]=(np.mean(obj[:,i]))
    
#     plt.figure()
#     plt.plot(range(50),objMean,"k-")
#     plt.fill_between(range(50), objMean-objSTD, objMean+objSTD,alpha=0.5, facecolor='b')
#     plt.title('Network Performance- Object Classification ({} realizations) Network Size: {}' .format(numberOfRealizations, nHiddSizeVec[netwSize]),fontsize=24)
#     plt.xlabel('leaky rate',fontsize=32)
#     plt.ylabel('Accuracy',fontsize=24)
#     plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
#     axes = plt.gca()
#     axes.set_ylim(0, 1.2)
#     plt.show()




# for netwSize in range(len(nHiddSizeVec)):      
#     widthSTD=np.empty(50)
#     width=resultsFixedWidth[:,:,netwSize]
#     for i in range(width.shape[1]):
#          widthSTD[i]=(np.std(width[:,i]))
    
#     widthMean=np.empty(50)
#     for i in range(width.shape[1]):
#          widthMean[i]=(np.mean(width[:,i]))
    
#     plt.figure()
#     plt.plot(range(50),widthMean,"k-")
#     plt.fill_between(range(50), widthMean-widthSTD, widthMean+widthSTD,alpha=0.5, facecolor='lightcoral')
#     plt.title('Width Classification ({} realizations) Network Size: {}' .format(numberOfRealizations, nHiddSizeVec[netwSize]),fontsize=24)
#     plt.xlabel('leaky rate',fontsize=32)
#     plt.ylabel('Accuracy',fontsize=24)
#     plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
#     axes = plt.gca()
#     axes.set_ylim(0, 1)
#     plt.show()


# #both plots
# plt.figure()
# plt.plot(range(50),widthMean,"k-")
# plt.fill_between(range(50), widthMean-widthSTD, widthMean+widthSTD,alpha=0.5, facecolor='lightcoral')
# plt.xlabel('leaky rate')
# plt.ylabel('Ratio Of correct responses to Number Of Images')
# plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
# plt.plot(range(50),objMean,"k-")
# plt.fill_between(range(50), objMean-objSTD, objMean+objSTD,alpha=0.5, facecolor='b')
# plt.title('Network Performance- Object Classification VS. Width Classification  (100 realizations)')
# plt.xlabel('leaky rate')
# plt.ylabel('Ratio Of correct responses to Number Of Images')
# plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 20}

# plt.rc('font', **font)
# plt.show()







# #plotting performance on top of each other
# for netwSize in range(len(nHiddSizeVec)):         
#     objSTD=np.empty(50)
#     obj=resultsRotObj[:,:,netwSize]
#     for i in range(obj.shape[1]):
#          objSTD[i]=(np.std(obj[:,i]))
         
#     objMean=np.empty(50)
#     for i in range(obj.shape[1]):
#          objMean[i]=(np.mean(obj[:,i]))
#     widthSTD=np.empty(50)
#     width=resultsFixedWidth[:,:,netwSize]
#     for i in range(width.shape[1]):
#          widthSTD[i]=(np.std(width[:,i]))
    
#     widthMean=np.empty(50)
#     for i in range(width.shape[1]):
#          widthMean[i]=(np.mean(width[:,i]))
    
#     plt.figure()
#     plt.plot(range(50),objMean,"k-")
#     plt.fill_between(range(50), objMean-objSTD, objMean+objSTD,alpha=0.5, facecolor='b',label='Object')
#     plt.plot(range(50),widthMean,"k-")
#     plt.fill_between(range(50), widthMean-widthSTD, widthMean+widthSTD,alpha=0.5, facecolor='lightcoral',label='Width')

#     plt.title('Network performance- ({} realizations) Network size: {}' .format(numberOfRealizations, nHiddSizeVec[netwSize]),fontsize=24)
#     plt.xlabel('Leaking rate',fontsize=24)
#     plt.ylabel('Accuracy',fontsize=24)
#     plt.xticks(ticks=[0,10, 20, 30, 40, 50 ],labels=[0, 0.20, 0.40, 0.60, 0.80, 1.0 ],fontsize=22)
#     plt.yticks(fontsize=22)
#     axes = plt.gca()
#     axes.set_ylim(0.4, 1.1)
#     # plt.legend()
#     font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 20}

#     plt.rc('font', **font)
#     plt.show()


# #end of plotting both
##################################################################################