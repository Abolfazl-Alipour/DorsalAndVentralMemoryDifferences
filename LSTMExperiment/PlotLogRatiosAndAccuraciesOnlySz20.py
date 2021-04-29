#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:05:10 2021

@author: aalipour
"""
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pickle

numOfDsets=3
numberOfRealizations=30
nHiddSizeVec=[20]#,40,80,160,320,640]
sizeID=1
gateList=[]

with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/Coil100OriTrainTest200Epoch10-4LerRate30deg5degwiggle.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMFixedWidth, cumFGateValsCoilOri, cumIGateValsCoilOri = pickle.load(f)
    f.close()
with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/Coil100ObjTrainTest200Epoch10-4LerRate30deg5degwiggle.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMRotObj, cumFGateValsCoilObj, cumIGateValsCoilObj = pickle.load(f)
    f.close()




objLogRatiosMean=np.empty(numOfDsets)
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(numOfDsets)
objLogRatiosSEM[:]=np.nan

eachRalizationMean=np.empty(numberOfRealizations)
eachRalizationMean[:]=np.nan

objLogRatiosMean=np.empty(len(nHiddSizeVec))
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(len(nHiddSizeVec))
objLogRatiosSEM[:]=np.nan

eachRalizationMean=np.empty(numberOfRealizations)
eachRalizationMean[:]=np.nan
for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsCoilObj[sizeID][jj])):
            tmpF=np.array(cumFGateValsCoilObj[sizeID][jj][kk])
            tmpI=np.array(cumIGateValsCoilObj[sizeID][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRalizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)

widthLogRatiosMean=np.empty(len(nHiddSizeVec))
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(len(nHiddSizeVec))
widthLogRatiosSEM[:]=np.nan        

coilOri=eachRalizationMean*1

for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsCoilOri[sizeID][jj])):
            tmpF=np.array(cumFGateValsCoilOri[sizeID][jj][kk])
            tmpI=np.array(cumIGateValsCoilOri[sizeID][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRalizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)

coilobjLogRatiosMean=objLogRatiosMean   
coilobjLogRatiosSEM=objLogRatiosSEM
coilwidthLogRatiosMean=widthLogRatiosMean
coilwidthLogRatiosSEM=widthLogRatiosSEM

coilObj=eachRalizationMean*1
stats.ttest_ind(coilOri, coilObj)
################################    if MIRO semantic


with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/MIROOriTrainTest200Epoch10-3LerRateMidSize.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMFixedWidth, cumFGateValsMIROOri, cumIGateValsMIROOri = pickle.load(f)
    f.close()
with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/MIROObjTrainTest200Epoch10-3LerRate30deg5degwiggle.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMRotObj, cumFGateValsMIROObj, cumIGateValsMIROObj = pickle.load(f)
    f.close()




objLogRatiosMean=np.empty(len(nHiddSizeVec))
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(len(nHiddSizeVec))
objLogRatiosSEM[:]=np.nan

eachRalizationMean=np.empty(numberOfRealizations)
eachRalizationMean[:]=np.nan
for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsMIROObj[sizeID][jj])):
            tmpF=np.array(cumFGateValsMIROObj[sizeID][jj][kk])
            tmpI=np.array(cumIGateValsMIROObj[sizeID][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRalizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)

widthLogRatiosMean=np.empty(len(nHiddSizeVec))
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(len(nHiddSizeVec))
widthLogRatiosSEM[:]=np.nan        


MIROOri=eachRalizationMean*1


for ii in range (len(nHiddSizeVec)): #each netwrk size

    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsMIROOri[sizeID][jj])):
            tmpF=np.array(cumFGateValsMIROOri[sizeID][jj][kk])
            tmpI=np.array(cumIGateValsMIROOri[sizeID][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRalizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)
    
MIROobjLogRatiosMean=objLogRatiosMean   
MIROobjLogRatiosSEM=objLogRatiosSEM
MIROwidthLogRatiosMean=widthLogRatiosMean
MIROwidthLogRatiosSEM=widthLogRatiosSEM    
    
MIROObj=eachRalizationMean*1
stats.ttest_ind(MIROOri, MIROObj)

################################    if MIRO Visual 
with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/MIROOriTrainTest200Epoch10-3LerRateMidSizeFG.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMFixedWidth, cumFGateValsMIROOri, cumIGateValsMIROOri = pickle.load(f)
    f.close()


with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/MIROObjTrainTest200Epoch10-3LerRateSpinerMidSizeFG.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMRotObj, cumFGateValsMIROObj, cumIGateValsMIROObj = pickle.load(f)
    f.close()



objLogRatiosMean=np.empty(len(nHiddSizeVec))
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(len(nHiddSizeVec))
objLogRatiosSEM[:]=np.nan

eachRalizationMean=np.empty(numberOfRealizations)
eachRalizationMean[:]=np.nan
for ii in range (len(nHiddSizeVec)): #each netwrk size

    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsMIROObj[sizeID][jj])):
            tmpF=np.array(cumFGateValsMIROObj[sizeID][jj][kk])
            tmpI=np.array(cumIGateValsMIROObj[sizeID][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRalizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)

widthLogRatiosMean=np.empty(len(nHiddSizeVec))
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(len(nHiddSizeVec))
widthLogRatiosSEM[:]=np.nan        

MIROFGOri=eachRalizationMean*1

for ii in range (len(nHiddSizeVec)): #each netwrk size

    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsMIROOri[sizeID][jj])):
            tmpF=np.array(cumFGateValsMIROOri[sizeID][jj][kk])
            tmpI=np.array(cumIGateValsMIROOri[sizeID][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRalizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)


MIROFGobjLogRatiosMean=objLogRatiosMean   
MIROFGobjLogRatiosSEM=objLogRatiosSEM
MIROFGwidthLogRatiosMean=widthLogRatiosMean
MIROFGwidthLogRatiosSEM=widthLogRatiosSEM    
    
    
MIROFGObj=eachRalizationMean*1
stats.ttest_ind(MIROFGOri, MIROFGObj)


#####################  PLOT NOW

objLogRatiosMean=[coilobjLogRatiosMean[0], MIROobjLogRatiosMean[0],MIROFGobjLogRatiosMean[0]]
objLogRatiosSEM=[coilobjLogRatiosSEM[0],MIROobjLogRatiosSEM[0], MIROFGobjLogRatiosSEM[0]]
widthLogRatiosMean=[coilwidthLogRatiosMean[0],MIROwidthLogRatiosMean[0], MIROFGwidthLogRatiosMean[0]]
widthLogRatiosSEM=[coilwidthLogRatiosSEM[0],MIROwidthLogRatiosSEM[0], MIROFGwidthLogRatiosSEM[0]]


labels = ['COIL-100', 'MIRO Semantic', 'MIRO Visual']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
myPal=sns.color_palette(['b','r'])
sns.set_theme()
sns.set_theme(context='notebook', style='darkgrid', palette=myPal)
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, objLogRatiosMean, width,yerr= objLogRatiosSEM,label='Object',capsize=5)
rects2 = ax.bar(x + width/2, widthLogRatiosMean, width,yerr=widthLogRatiosSEM, label='Orientation',capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('LSTM Memory Coefficient',fontsize=24)
# ax.set_xlabel('Network Size',fontsize=24)
# ax.set_ylim([-0.5,0.2])
ax.set_title('LSTM Memory Coefficient across different network sizes',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26,loc='upper center')
# fig.tight_layout()



plt.show()
    





