#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 20:51:19 2021

@author: aalipour
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:42:51 2020

@author: aalipour
"""

#load the width and obj data first

#average logs across 30 realizations USED FOR PAPER
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import pickle
import scipy.stats as stats

nHiddSizeVec=[20]#,40,80,160,320,640]
# nHiddSizeVec=[160]#,40,80,160,320,640]
ii=0
numberOfRealizations=30
numOfEpochs=20



################################    if coil



ii=0
##Getting back the objects:
with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/Coil100OriTrainTest200Epoch10-4LerRate30deg5degwiggleAllEpchsSize='+ str(nHiddSizeVec[ii])+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMFixedWidth, cumFGateValsCoilOri, cumIGateValsCoilOri = pickle.load(f)


with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/Coil100ObjTrainTest200Epoch10-4LerRate30deg5degwiggleAllEpchsSize='+ str(nHiddSizeVec[ii])+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMRotObj, cumFGateValsCoilObj, cumIGateValsCoilObj = pickle.load(f)

accResultLSTMFixedWidth=accResultLSTMFixedWidth[0]
cumFGateValsCoilOri=cumFGateValsCoilOri[0]
cumIGateValsCoilOri=cumIGateValsCoilOri[0]

accResultLSTMRotObj=accResultLSTMRotObj[0]
cumFGateValsCoilObj=cumFGateValsCoilObj[0]
cumIGateValsCoilObj=cumIGateValsCoilObj[0]

objLogRatiosMean=np.empty(numOfEpochs)
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(numOfEpochs)
objLogRatiosSEM[:]=np.nan

eachRealizationMean=np.empty(numberOfRealizations)
eachRealizationMean[:]=np.nan
for ii in range (numOfEpochs): #each realization
    for jj in range (numberOfRealizations):  #each eppch
        tmp=[]
        for kk in range(len(cumFGateValsCoilObj[jj][ii])):
            tmpF=np.array(cumFGateValsCoilObj[jj][ii][kk])
            tmpI=np.array(cumIGateValsCoilObj[jj][ii][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
            # tmp.append(np.mean(np.ma.masked_invalid(tmpF/tmpI)))
        eachRealizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRealizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRealizationMean)

coilObj=eachRealizationMean

widthLogRatiosMean=np.empty(numOfEpochs)
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(numOfEpochs)
widthLogRatiosSEM[:]=np.nan        

eachRealizationMean=np.empty(numberOfRealizations)
eachRealizationMean[:]=np.nan

for ii in range (numOfEpochs): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsCoilOri[jj][ii])):
            tmpF=np.array(cumFGateValsCoilOri[jj][ii][kk])
            tmpI=np.array(cumIGateValsCoilOri[jj][ii][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
            # tmp.append(np.mean(np.ma.masked_invalid(tmpF/tmpI)))
        eachRealizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRealizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRealizationMean)

coilOri=eachRealizationMean
stats.ttest_ind(coilOri, coilObj)

colorsObj = ("cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","royalblue","royalblue","royalblue","royalblue","b","b","b","mediumblue","mediumblue","darkblue","k")
colorsWidth=("lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","indianred","indianred","indianred","indianred","brown","brown","brown","firebrick","firebrick","maroon","k")




# Create plot
sns.set_theme( )
# sns.axes_style("whitegrid")
markerSize=150
fig = plt.figure()
#plt.title('Leaky Rate VS. Classification Accuracy')

ax = fig.add_subplot(1, 1, 1)
# ax=plt.plot()
ax.tick_params(labelsize=22) 
ax.scatter(np.mean(accResultLSTMRotObj,axis=0)*100, objLogRatiosMean, alpha=0.8, c=colorsObj, edgecolors='none', s=markerSize,label='Object classification Mem. Coef.')
ax.scatter(np.mean(accResultLSTMFixedWidth,axis=0)*100, widthLogRatiosMean, alpha=0.8, c=colorsWidth, edgecolors='none', s=markerSize,label='Orientation classification Mem. Coef.')
# ax.set_yscale('log')

# ax.scatter(0.2,.75, alpha=0.8, marker='>',c='k', edgecolors='none', s=300,label='Initial Mem. Coef.')
ax.legend(fontsize=26)

ax.set_xlabel('Network Accuracy (%)',size=24)
ax.set_ylabel('Memory Coefficient',size=24)

ax.set_ylim(-.2, 1)
ax.set_xlim(0, 110)


coilObjRtio=objLogRatiosMean[-1]
coilObjSEM=objLogRatiosSEM[-1]
coilOriRtio=widthLogRatiosMean[-1]
coilOriSEM=widthLogRatiosSEM[-1]




################################    if MIRO Semantic (incongruent)
ii=0

with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/MIROOriTrainTest200Epoch10-3LerRateMidSizeAllEpchsSize='+ str(nHiddSizeVec[ii])+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMFixedWidthMIROSEMNTC, cumFGateValsMIROOri, cumIGateValsMIROOri = pickle.load(f)
    
with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/MIROObjTrainTest200Epoch10-3LerRateSpinerMidSizeAllEpchsSize='+ str(nHiddSizeVec[ii])+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMRotObjINCON, cumFGateValsMIROObjINCON, cumIGateValsMIROObjINCON = pickle.load(f)



accResultLSTMFixedWidthMIROSEMNTC=accResultLSTMFixedWidthMIROSEMNTC[0]
cumFGateValsMIROOri=cumFGateValsMIROOri[0]
cumIGateValsMIROOri=cumIGateValsMIROOri[0]

accResultLSTMRotObjINCON=accResultLSTMRotObjINCON[0]
cumFGateValsMIROObjINCON=cumFGateValsMIROObjINCON[0]
cumIGateValsMIROObjINCON=cumIGateValsMIROObjINCON[0]

objLogRatiosMean=np.empty(numOfEpochs)
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(numOfEpochs)
objLogRatiosSEM[:]=np.nan

eachRealizationMean=np.empty(numberOfRealizations)
eachRealizationMean[:]=np.nan
for ii in range (numOfEpochs): #each realization
    for jj in range (numberOfRealizations):  #each eppch
        tmp=[]
        for kk in range(len(cumFGateValsMIROObjINCON[jj][ii])):
            tmpF=np.array(cumFGateValsMIROObjINCON[jj][ii][kk])
            tmpI=np.array(cumIGateValsMIROObjINCON[jj][ii][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
            # tmp.append(np.mean(np.ma.masked_invalid(tmpF/tmpI)))
        eachRealizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRealizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRealizationMean)
    
MIROObj=eachRealizationMean

widthLogRatiosMean=np.empty(numOfEpochs)
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(numOfEpochs)
widthLogRatiosSEM[:]=np.nan        

eachRealizationMean=np.empty(numberOfRealizations)
eachRealizationMean[:]=np.nan

for ii in range (numOfEpochs): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsMIROOri[jj][ii])):
            tmpF=np.array(cumFGateValsMIROOri[jj][ii][kk])
            tmpI=np.array(cumIGateValsMIROOri[jj][ii][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
            # tmp.append(np.mean(np.ma.masked_invalid(tmpF/tmpI)))
        eachRealizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRealizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRealizationMean)


### exception since it gets to the 10% on epoch 2
for ii in range (2): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsMIROOri[jj][ii])):
            tmpF=np.array(cumFGateValsMIROOri[jj][ii][kk])
            tmpI=np.array(cumIGateValsMIROOri[jj][ii][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
            # tmp.append(np.mean(np.ma.masked_invalid(tmpF/tmpI)))
        eachRealizationMean[jj]=np.mean(np.array(tmp))
        
MIROOri=eachRealizationMean
stats.ttest_ind(MIROOri, MIROObj)


# full version
colorsObj = ("cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","royalblue","royalblue","royalblue","royalblue","b","b","b","mediumblue","mediumblue","darkblue","k")
colorsWidth=("lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","indianred","indianred","indianred","indianred","brown","brown","brown","firebrick","firebrick","maroon","k")




# Create plot
sns.set_theme( )
# sns.axes_style("whitegrid")
markerSize=150
fig = plt.figure()
#plt.title('Leaky Rate VS. Classification Accuracy')

ax = fig.add_subplot(1, 1, 1)
# ax=plt.plot()
ax.tick_params(labelsize=22) 
ax.scatter(np.mean(accResultLSTMRotObjINCON,axis=0)*100, objLogRatiosMean, alpha=0.8, c=colorsObj, edgecolors='none', s=markerSize,label='Object classification Mem. Coef.')
ax.scatter(np.mean(accResultLSTMFixedWidthMIROSEMNTC,axis=0)*100, widthLogRatiosMean, alpha=0.8, c=colorsWidth, edgecolors='none', s=markerSize,label='Orientation classification Mem. Coef.')

# ax.set_yscale('log')

# ax.scatter(0.2,.75, alpha=0.8, marker='>',c='k', edgecolors='none', s=300,label='Initial Mem. Coef.')
ax.legend(fontsize=30)

ax.set_xlabel('Network Accuracy (%)',size=24)
ax.set_ylabel('Memory Coefficient',size=24)

ax.set_ylim(-.2, 1)
ax.set_xlim(0, 110)

MIROObjRtio=objLogRatiosMean[-1]
MIROObjSEM=objLogRatiosSEM[-1]
MIROOriRtio=widthLogRatiosMean[1]
MIROOriSEM=widthLogRatiosSEM[1]

# just to 100% version
colorsObj = ("cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","royalblue","royalblue","royalblue","royalblue","b","b","b","mediumblue","mediumblue","darkblue","k")
colorsWidth=("lightcoral","maroon")




# Create plot
sns.set_theme( )
# sns.axes_style("whitegrid")
markerSize=150
fig = plt.figure()
#plt.title('Leaky Rate VS. Classification Accuracy')

ax = fig.add_subplot(1, 1, 1)
# ax=plt.plot()
ax.tick_params(labelsize=22) 
ax.scatter(np.mean(accResultLSTMRotObjINCON,axis=0)*100, objLogRatiosMean, alpha=0.8, c=colorsObj, edgecolors='none', s=markerSize,label='Object classification Mem. Coef.')
ax.scatter(np.mean(accResultLSTMFixedWidthMIROSEMNTC[:,0:2],axis=0)*100, widthLogRatiosMean[0:2], alpha=0.8, c=colorsWidth, edgecolors='none', s=markerSize,label='Orientation classification Mem. Coef.')
# ax.set_yscale('log')

# ax.scatter(0.2,.75, alpha=0.8, marker='>',c='k', edgecolors='none', s=300,label='Initial Mem. Coef.')
# ax.legend(fontsize=30)

ax.set_xlabel('Network Accuracy (%)',size=24)
ax.set_ylabel('Memory Coefficient',size=24)

ax.set_ylim(-.2, 1)
ax.set_xlim(0, 110)



################################    if MIRO FG (visual)



ii=0
with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/MIROOriTrainTest200Epoch10-3LerRateMidSizeFGAllEpchsSize='+ str(nHiddSizeVec[ii])+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMFixedWidthMIROFG, cumFGateValsMIROOriFG, cumIGateValsMIROOriFG = pickle.load(f)


with open('/N/u/aalipour/Carbonate/Downloads/2ndYearproject/2020Wave/code/ReviewersAsked/code/MIROObjTrainTest200Epoch10-3LerRateSpinerMidSizeFGAllEpchsSize='+ str(nHiddSizeVec[ii])+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
    accResultLSTMRotObjFG, cumFGateValsMIROObjFG, cumIGateValsMIROObjFG = pickle.load(f)

accResultLSTMFixedWidthMIROFG=accResultLSTMFixedWidthMIROFG[0]
cumFGateValsMIROOriFG=cumFGateValsMIROOriFG[0]
cumIGateValsMIROOriFG=cumIGateValsMIROOriFG[0]

accResultLSTMRotObjFG=accResultLSTMRotObjFG[0]
cumFGateValsMIROObjFG=cumFGateValsMIROObjFG[0]
cumIGateValsMIROObjFG=cumIGateValsMIROObjFG[0]

objLogRatiosMean=np.empty(numOfEpochs)
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(numOfEpochs)
objLogRatiosSEM[:]=np.nan

eachRealizationMean=np.empty(numberOfRealizations)
eachRealizationMean[:]=np.nan
for ii in range (numOfEpochs): #each realization
    for jj in range (numberOfRealizations):  #each eppch
        tmp=[]
        for kk in range(len(cumFGateValsMIROObjFG[jj][ii])):
            tmpF=np.array(cumFGateValsMIROObjFG[jj][ii][kk])
            tmpI=np.array(cumIGateValsMIROObjFG[jj][ii][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
            # tmp.append(np.mean(np.ma.masked_invalid(tmpF/tmpI)))
        eachRealizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRealizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRealizationMean)

MIROFGObj=eachRealizationMean


widthLogRatiosMean=np.empty(numOfEpochs)
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(numOfEpochs)
widthLogRatiosSEM[:]=np.nan        

eachRealizationMean=np.empty(numberOfRealizations)
eachRealizationMean[:]=np.nan

for ii in range (numOfEpochs): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumIGateValsMIROOriFG[jj][ii])):
            tmpF=np.array(cumFGateValsMIROOriFG[jj][ii][kk])
            tmpI=np.array(cumIGateValsMIROOriFG[jj][ii][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
            # tmp.append(np.mean(np.ma.masked_invalid(tmpF/tmpI)))
        eachRealizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRealizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRealizationMean)


MIROFGOri=eachRealizationMean
stats.ttest_ind(MIROFGOri, MIROFGObj)

colorsObj = ("cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","royalblue","royalblue","royalblue","royalblue","b","b","b","mediumblue","mediumblue","darkblue","k")
colorsWidth=("lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","indianred","indianred","indianred","indianred","brown","brown","brown","firebrick","firebrick","maroon","k")




# Create plot
sns.set_theme( )
# sns.axes_style("whitegrid")
markerSize=150
fig = plt.figure()
#plt.title('Leaky Rate VS. Classification Accuracy')

ax = fig.add_subplot(1, 1, 1)
# ax=plt.plot()
ax.tick_params(labelsize=22) 
ax.scatter(np.mean(accResultLSTMRotObjFG,axis=0)*100, objLogRatiosMean, alpha=0.8, c=colorsObj, edgecolors='none', s=markerSize,label='Object classification Mem. Coef.')
ax.scatter(np.mean(accResultLSTMFixedWidthMIROFG,axis=0)*100, widthLogRatiosMean, alpha=0.8, c=colorsWidth, edgecolors='none', s=markerSize,label='Orientation classification Mem. Coef.')
# ax.set_yscale('log')

# ax.scatter(0.2,.75, alpha=0.8, marker='>',c='k', edgecolors='none', s=300,label='Initial Mem. Coef.')
ax.legend(fontsize=30,loc='upper left')

ax.set_xlabel('Network Accuracy (%)',size=24)
ax.set_ylabel('Memory Coefficient',size=24)

ax.set_ylim(-.2, 1)
ax.set_xlim(0, 110)



MIROFGObjRtio=objLogRatiosMean[-1]
MIROFGObjSEM=objLogRatiosSEM[-1]
MIROFGOriRtio=widthLogRatiosMean[-1]
MIROFGOriSEM=widthLogRatiosSEM[-1]




objLogRatiosMean=[coilObjRtio,MIROFGObjRtio,MIROObjRtio]
objLogRatiosSEM=[coilObjSEM,MIROFGObjSEM,MIROObjSEM]
widthLogRatiosMean=[coilOriRtio,MIROFGOriRtio,MIROOriRtio]
widthLogRatiosSEM=[coilOriSEM,MIROFGOriSEM,MIROOriSEM]


labels = ['COIL-100', 'MIRO Visual', 'MIRO Semantic']

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
# ax.set_title('LSTM Memory Coefficient across different network sizes',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26,loc='upper right')

ax.set_ylim(0,1)
# fig.tight_layout()



plt.show()
    