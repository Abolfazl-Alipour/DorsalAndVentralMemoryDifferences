# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:38:30 2020

@author: aalipour
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
nHiddSizeVec=[20,40,80,160,320,640]
for kk in range(WidthMeanFGate.shape[0]):
    WidthnumberOfRealizations=WidthMeanFGate.shape[1]
    WidthNumOfEpochs=WidthMeanFGate.shape[2]
    widthFGateSEM=np.empty(WidthNumOfEpochs)
    widthFGateSEM[:]=np.nan
    widthIGateSEM=np.empty(WidthNumOfEpochs)
    widthIGateSEM[:]=np.nan
    widthaccMean=np.empty(WidthNumOfEpochs)
    widthaccMean[:]=np.nan
    widthFandIDiff=np.empty(WidthNumOfEpochs)
    widthFandIDiff[:]=np.nan
    
    
    for i in range(WidthMeanFGate.shape[2]):
         widthFGateSEM[i]=(scipy.stats.sem(WidthMeanFGate[kk,:,i]))
    
    widthFGateMean=np.empty(WidthNumOfEpochs)
    for i in range(WidthMeanFGate.shape[2]):
         widthFGateMean[i]=(np.mean(WidthMeanFGate[kk,:,i]))
         
         
    for i in range(accResultLSTMFixedWidth.shape[2]):
        widthaccMean[i]=(np.mean(accResultLSTMFixedWidth[kk,:,i]))
         
    
    for i in range(WidthMeanIGate.shape[2]):
         widthIGateSEM[i]=(scipy.stats.sem(WidthMeanIGate[kk,:,i]))
         
    
    widthIGateMean=np.empty(WidthNumOfEpochs)
    for i in range(WidthMeanIGate.shape[2]):
         widthIGateMean[i]=(np.mean(WidthMeanIGate[kk,:,i]))
         
    for i in range(WidthFandIDiff.shape[2]):
         widthFandIDiff[i]=(np.mean(WidthFandIDiff[0,:,i]))
     
    #for i in range(accResultsLSTMRotObj.shape[2]):
    #     RotObjaccMean[i]=(np.mean(accResultsLSTMRotObj[0,:,i]))
    
    
    # width and obj recog
    
    
    fig = plt.figure()
    
    plt.fill_between(range(WidthNumOfEpochs), widthFGateMean-widthFGateSEM, widthFGateMean+widthFGateSEM,alpha=0.5, facecolor='lightcoral',label='Forget Gate')
    plt.plot(range(WidthNumOfEpochs),widthFGateMean,"k-")
    plt.fill_between(range(WidthNumOfEpochs), widthIGateMean-widthIGateSEM, widthIGateMean+widthIGateSEM,alpha=0.5, facecolor='b',label='Input Gate')
    plt.plot(range(WidthNumOfEpochs),widthIGateMean,"k-")
    plt.plot(range(WidthNumOfEpochs),widthFandIDiff,"k-",label='F-I Gate')
    plt.title('Mean Forget and Input Gate during Orientation Detectoin learning- ({} realizations) Network Size: {}' .format(WidthnumberOfRealizations, nHiddSizeVec[kk]),fontsize=24)#nHiddSizeVec[netwSize]
    plt.xlabel('Accuracy (range= 40 Epoch)',fontsize=32)
    plt.ylabel('Mean Value',fontsize=24)
    tmp=(widthaccMean[::10]*100).tolist()
    tmp.append(widthaccMean[-1]*100)
    xlabels=np.round(tmp)
    plt.xticks(ticks=[0, 10, 20, 30,39],labels=(xlabels))
    axes = plt.gca()
#    axes.set_ylim(.45, .9)
    plt.legend(loc='upper left',fontsize=24)
    plt.show()

for kk in range(ObjMeanFGate.shape[0]):
    objNumberOfRealizations=ObjMeanFGate.shape[1]
    objNumOfEpochs=ObjMeanFGate.shape[2]
    RotObjthnumberOfRealizations=ObjMeanFGate.shape[1]
    RotObjNumOfEpochs=ObjMeanFGate.shape[2]
    RotObjFGateSEM=np.empty(RotObjNumOfEpochs)
    RotObjFGateSEM[:]=np.nan
    RotObjIGateSEM=np.empty(RotObjNumOfEpochs)
    RotObjIGateSEM[:]=np.nan
    RotObjaccMean=np.empty(RotObjNumOfEpochs)
    RotObjaccMean[:]=np.nan
    RotObjFandIDIff=np.empty(RotObjNumOfEpochs)
    RotObjFandIDIff[:]=np.nan
    
    
    for i in range(ObjMeanFGate.shape[2]):
         RotObjFGateSEM[i]=(scipy.stats.sem(ObjMeanFGate[0,:,i]))
    
    RotObjFGateMean=np.empty(RotObjNumOfEpochs)
    for i in range(ObjMeanFGate.shape[2]):
         RotObjFGateMean[i]=(np.mean(ObjMeanFGate[0,:,i]))
         
         
    for i in range(accResultLSTMRotObj.shape[2]):
        RotObjaccMean[i]=(np.mean(accResultLSTMRotObj[0,:,i]))
         
    
    for i in range(ObjMeanIGate.shape[2]):
         RotObjIGateSEM[i]=(scipy.stats.sem(ObjMeanIGate[0,:,i]))
         
    
    RotObjIGateMean=np.empty(objNumOfEpochs)
    for i in range(ObjMeanIGate.shape[2]):
         RotObjIGateMean[i]=(np.mean(ObjMeanIGate[0,:,i]))
    
    for i in range(ObjFandIDiff.shape[2]):
         RotObjFandIDIff[i]=(np.mean(ObjFandIDiff[0,:,i]))
     
    
    
    fig = plt.figure()
    
    plt.fill_between(range(objNumOfEpochs), RotObjFGateMean-RotObjFGateSEM, RotObjFGateMean+RotObjFGateSEM,alpha=0.5, facecolor='lightcoral',label='Forget Gate')
    plt.plot(range(objNumOfEpochs),RotObjFGateMean,"k-")
    plt.fill_between(range(objNumOfEpochs), RotObjIGateMean-RotObjIGateSEM, RotObjIGateMean+RotObjIGateSEM,alpha=0.5, facecolor='b',label='Input Gate')
    plt.plot(range(objNumOfEpochs),RotObjIGateMean,"k-")
    plt.plot(range(objNumOfEpochs),RotObjFandIDIff,"k-")
    plt.title('Mean Forget and Input Gate during Obj Recognition learning- ({} realizations) Network Size: {}' .format(objNumberOfRealizations, nHiddSizeVec[kk]),fontsize=24)#nHiddSizeVec[netwSize]
    plt.xlabel('Accuracy (range= 10 Epoch)',fontsize=32)
    plt.ylabel('Mean Value',fontsize=24)
    tmp=(RotObjaccMean*100).tolist()
    xlabels=np.round(tmp)
    plt.xticks(ticks=range(10),labels=(xlabels))
    axes = plt.gca()
#    axes.set_ylim(.45, .9)
    plt.legend(loc='upper left',fontsize=24)
    plt.show()




# width and obj recog
import matplotlib.pyplot as plt

fig = plt.figure()

plt.fill_between(range(WidthNumOfEpochs), widthFGateMean-widthFGateSEM, widthFGateMean+widthFGateSEM,alpha=0.5, facecolor='lightcoral')
plt.fill_between(range(WidthNumOfEpochs), widthIGateMean-widthSTD, widthIGateMean+widthSTD,alpha=0.5, facecolor='b')








plt.fill_between(range(10), MeanFGate[0,0,:]-widthSTD, MeanFGate[0,0,:]+widthSTD,alpha=0.5, facecolor='lightcoral')
plt.fill_between(range(10), MeanIGate[0,0,:]-widthSTD, MeanIGate[0,0,:]+widthSTD,alpha=0.5, facecolor='b')




#plt.title('Changes in Forget Gate to Input Gate Ratio during network training ({} realizations)' .format(numberOfRealizations),fontsize=24)
#plt.xlabel('Accuracy After Each training Epoch',fontsize=32,va='bottom')


ax1.set_xticks(xTicks)
ax1.set_xticklabels((np.around( widthaccMean*100,decimals=0))[0:10],fontsize=20,color='r')
ax2.set_xticks(xTicks)
ax2.set_xticklabels((np.around( RotObjaccMean*100,decimals=0 )),fontsize=20,color='b')
ax1.set_xlabel('Accuracy After Each training Epoch',fontsize=24)#,va='bottom')
ax1.set_ylabel('Mean Forget Gate to Input Gate Ratio',fontsize=24)
matplotlib.rc('ytick', labelsize=32) 
axes = plt.gca()
#axes.set_ylim(float(10**33),float(6*(10**33)))

plt.show()