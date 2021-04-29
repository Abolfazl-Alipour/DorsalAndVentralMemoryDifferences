#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:38:58 2020

@author: aalipour
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:31:17 2020

@author: aalipour
"""



import torch
import numpy as np
import scipy.stats
import pickle
import matplotlib.pyplot as plt
from loadPretrainedOriclassifierAndIncorporateitintoESNForRotatingObjMIROFinegrain import ESNRotatingObject
from loadPretrainedOriclassifierAndIncorporateitintoESNForFixedWidthMIROFinegrain import ESNFixedWidth
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
with open('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/ESNWidthAllLeakingRatesMIROMIDSIZEFinegrain.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([resultsFixedWidth], f)
    # f.close()
# # Getting back the objects:
# with open('/geode2/home/u010/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/ESNWidthAllLeakingRatesMIROMIDSIZEFinegrain.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     resultsFixedWidth = pickle.load(f)
