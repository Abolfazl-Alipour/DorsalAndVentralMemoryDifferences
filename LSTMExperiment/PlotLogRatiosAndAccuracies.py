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


nHiddSizeVec=[20,40,80,160,320,640]
numberOfRealizations=30


# if synthetic:
    
objLogRatiosMean=np.empty(len(nHiddSizeVec))
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(len(nHiddSizeVec))
objLogRatiosSEM[:]=np.nan

eachRalizationMean=np.empty(numberOfRealizations)
eachRalizationMean[:]=np.nan
for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsSynthObj[ii][jj])):
            tmpF=np.array(cumFGateValsSynthObj[ii][jj][kk])
            tmpI=np.array(cumIGateValsSynthObj[ii][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRalizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)


widthLogRatiosMean=np.empty(len(nHiddSizeVec))
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(len(nHiddSizeVec))
widthLogRatiosSEM[:]=np.nan        


for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsSynthOri[ii][jj])):
            tmpF=np.array(cumFGateValsSynthOri[ii][jj][kk])
            tmpI=np.array(cumIGateValsSynthOri[ii][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRalizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)

################################    if coil

objLogRatiosMean=np.empty(len(nHiddSizeVec))
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(len(nHiddSizeVec))
objLogRatiosSEM[:]=np.nan

eachRalizationMean=np.empty(numberOfRealizations)
eachRalizationMean[:]=np.nan
for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsCoilObj[ii][jj])):
            tmpF=np.array(cumFGateValsCoilObj[ii][jj][kk])
            tmpI=np.array(cumIGateValsCoilObj[ii][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRalizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)

widthLogRatiosMean=np.empty(len(nHiddSizeVec))
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(len(nHiddSizeVec))
widthLogRatiosSEM[:]=np.nan        



for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsCoilOri[ii][jj])):
            tmpF=np.array(cumFGateValsCoilOri[ii][jj][kk])
            tmpI=np.array(cumIGateValsCoilOri[ii][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRalizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)

################################    if MIRO

objLogRatiosMean=np.empty(len(nHiddSizeVec))
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(len(nHiddSizeVec))
objLogRatiosSEM[:]=np.nan

eachRalizationMean=np.empty(numberOfRealizations)
eachRalizationMean[:]=np.nan
for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsMIROObj[ii][jj])):
            tmpF=np.array(cumFGateValsMIROObj[ii][jj][kk])
            tmpI=np.array(cumIGateValsMIROObj[ii][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRalizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)

widthLogRatiosMean=np.empty(len(nHiddSizeVec))
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(len(nHiddSizeVec))
widthLogRatiosSEM[:]=np.nan        



for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsMIROOri[ii][jj])):
            tmpF=np.array(cumFGateValsMIROOri[ii][jj][kk])
            tmpI=np.array(cumIGateValsMIROOri[ii][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRalizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)




############################ now plotting

labels = ['20', '40', '80', '160', '320','640']

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
ax.set_xlabel('Network Size',fontsize=24)
# ax.set_ylim([-0.5,0.2])
ax.set_title('LSTM Memory Coefficient across different network sizes',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26,loc='upper right')
# fig.tight_layout()



plt.show()



#######################################


# accuracies
##################  for Synthetic Dataset
lowerLim=70
higherLim=105
yticks=[70,80,90,100]
ylabels = ['70', '80','90', '100']
#################   for COIL-100 Dataset
lowerLim=50
higherLim=120
yticks=[60,70,80,90,100]
ylabels = ['60','70', '80', '90', '100']
###############################################################################

widthAccSTD=np.empty(6)
widthAccSTD[:]=np.nan
for i in range(accResultLSTMFixedWidth.shape[0]):
     widthAccSTD[i]=(np.std(accResultLSTMFixedWidth[i,:]))

widthAccMean=np.empty(6)
for i in range(accResultLSTMFixedWidth.shape[0]):
     widthAccMean[i]=(np.mean(accResultLSTMFixedWidth[i,:]))
    
    

ObjAccSTD=np.empty(6)
ObjAccSTD[:]=np.nan

for i in range(accResultLSTMRotObj.shape[0]):
     ObjAccSTD[i]=(np.std(accResultLSTMRotObj[i,:]))

ObjAccMean=np.empty(6)
for i in range(accResultLSTMRotObj.shape[0]):
     ObjAccMean[i]=(np.mean(accResultLSTMRotObj[i,:]))
     
     



xlabels = ['20', '40', '80', '160', '320','640']



x = np.arange(len(xlabels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ObjAccMean*100, width,yerr= ObjAccSTD*100,label='Object',color='blue',capsize=5)
rects2 = ax.bar(x + width/2, widthAccMean*100, width,yerr=widthAccSTD*100, label='Orientation',color='lightcoral',capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Test Accuracy (%)',fontsize=24)
ax.set_xlabel('Network Size',fontsize=24)
ax.set_title('Test accuracy across different network sizes',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(xlabels,fontsize=24)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26)
ax.set_ylim(lowerLim, higherLim)
#fig.tight_layout()
#plt.grid(which='minor')


plt.show()







########################### Plotting feature incongruent vs Fine grain congruent

objLogRatiosMean=np.empty(len(nHiddSizeVec))
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(len(nHiddSizeVec))
objLogRatiosSEM[:]=np.nan

eachRalizationMean=np.empty(numberOfRealizations)
eachRalizationMean[:]=np.nan
for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsMIROObjFG[ii][jj])):
            tmpF=np.array(cumFGateValsMIROObjFG[ii][jj][kk])
            tmpI=np.array(cumIGateValsMIROObjFG[ii][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRalizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)

widthLogRatiosMean=np.empty(len(nHiddSizeVec))
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(len(nHiddSizeVec))
widthLogRatiosSEM[:]=np.nan        



for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsMIROObjINCON[ii][jj])):
            tmpF=np.array(cumFGateValsMIROObjINCON[ii][jj][kk])
            tmpI=np.array(cumIGateValsMIROObjINCON[ii][jj][kk])
            tmp.append(np.mean(np.log10(np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRalizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)




############################ now plotting

labels = ['20', '40', '80', '160', '320','640']

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
ax.set_xlabel('Network Size',fontsize=24)
# ax.set_ylim([-0.5,0.2])
ax.set_title('LSTM Memory Coefficient across different network sizes',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26,loc='upper right')
# fig.tight_layout()



plt.show()
























#average logs across 30 realizations USED FOR PAPER
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
nHiddSizeVec=[20,40,80,160,320,640]
numberOfRealizations=30




objLogRatiosMean=np.empty(len(nHiddSizeVec))
objLogRatiosMean[:]=np.nan
objLogRatiosSEM=np.empty(len(nHiddSizeVec))
objLogRatiosSEM[:]=np.nan

eachRalizationMean=np.empty(numberOfRealizations)
eachRalizationMean[:]=np.nan
for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsSynthObj[ii][jj])):
            tmpF=np.array(cumFGateValsSynthObj[ii][jj][kk])
            tmpI=np.array(cumIGateValsSynthObj[ii][jj][kk])
            tmp.append(np.mean((np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    objLogRatiosMean[ii]=np.mean(eachRalizationMean)
    objLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)




widthLogRatiosMean=np.empty(len(nHiddSizeVec))
widthLogRatiosMean[:]=np.nan
widthLogRatiosSEM=np.empty(len(nHiddSizeVec))
widthLogRatiosSEM[:]=np.nan        


eachRalizationMean=np.empty(numberOfRealizations)
eachRalizationMean[:]=np.nan

for ii in range (len(nHiddSizeVec)): #each netwrk size
    for jj in range (numberOfRealizations):  #each realization
        tmp=[]
        for kk in range(len(cumFGateValsSynthOri[ii][jj])):
            tmpF=np.array(cumFGateValsSynthOri[ii][jj][kk])
            tmpI=np.array(cumIGateValsSynthOri[ii][jj][kk])
            tmp.append(np.mean((np.ma.masked_invalid(tmpF/tmpI))))
    
        eachRalizationMean[jj]=np.mean(np.array(tmp))
    widthLogRatiosMean[ii]=np.mean(eachRalizationMean)
    widthLogRatiosSEM[ii]=scipy.stats.sem(eachRalizationMean)



labels = ['20', '40', '80', '160', '320','640']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, objLogRatiosMean, width,yerr= objLogRatiosSEM,label='Object',color='blue',capsize=5)
rects2 = ax.bar(x + width/2, widthLogRatiosMean, width,yerr=widthLogRatiosSEM, label='Width',color='lightcoral',capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Log (Forget Gate / Input Gate)',fontsize=24)
ax.set_xlabel('Network Size',fontsize=24)
ax.set_title('Forget to input gate ratios across different network sizes',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26)
fig.tight_layout()



plt.show()




















nHiddSizeVec=[20,40,80,160,320,640]
numberOfRealizations=30


logMeanWidth=np.empty(len(nHiddSizeVec))
logMeanWidth[:]=np.nan
MeanWidth=logMeanWidth
widthRatioMeans=[]
widthRatioSEMs=[]

for ii in range(len(nHiddSizeVec)):

    FGateWidth=np.reshape(cumFGateValsSynthOri[ii,:,:,:,:],([numberOfRealizations,-1]))
    IGateWidth=np.reshape(cumIGateValsSynthOri[ii,:,:,:,:],([numberOfRealizations,-1]))
    logRatioWidth=np.log(np.ma.masked_invalid((FGateWidth)/(IGateWidth)))
    logRatioWidthMean=np.nanmean(logRatioWidth)
    logRatioWidthSEM=scipy.stats.sem(np.nanmean(logRatioWidth,axis=1),nan_policy='omit')
    

    widthRatioMeans.append(logRatioWidthMean)
    widthRatioSEMs.append(logRatioWidthSEM)
    
    
    
logMeanObj=np.empty(len(nHiddSizeVec))
logMeanObj[:]=np.nan
MeanObj=logMeanObj
objRatioMeans=[]
objRatioSEMs=[]
# note that some I Gate values are zero and you will see the runtime warning divide by zero ...
# we are ignoring those values in here
for ii in range(len(nHiddSizeVec)):
    FGateObj=np.reshape(cumFGateValsSynthObj[ii,:,:,:,:],([numberOfRealizations,-1]))
    IGateObj=np.reshape(cumIGateValsSynthObj[ii,:,:,:,:],([numberOfRealizations,-1]))
    logRatioObj=np.log(np.ma.masked_invalid((FGateObj)/(IGateObj)))
    logRatioObjMean=np.nanmean(logRatioObj)
    logRatioObjSEM=scipy.stats.sem(np.nanmean(logRatioObj,axis=1),nan_policy='omit')
    
   
    objRatioMeans.append(logRatioObjMean)
    objRatioSEMs.append(logRatioObjSEM)



labels = ['20', '40', '80', '160', '320','640']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, objRatioMeans, width,yerr= objRatioSEMs,label='Object',color='blue',capsize=5)
rects2 = ax.bar(x + width/2, widthRatioMeans, width,yerr=widthRatioSEMs, label='Width',color='lightcoral',capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Log (Forget Gate / Input Gate)',fontsize=24)
ax.set_xlabel('Network Size',fontsize=24)
ax.set_title('Forget to input gate ratios across different network sizes',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26)
fig.tight_layout()



plt.show()





#load the width and obj data first

#average logs across 30 realizations USED FOR PAPER
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
nHiddSizeVec=[20,40,80,160,320,640]
numberOfRealizations=30

logMeanWidth=np.empty(len(nHiddSizeVec))
logMeanWidth[:]=np.nan
MeanWidth=logMeanWidth
widthRatioMeans=[]
widthRatioSEMs=[]

for ii in range(len(nHiddSizeVec)):

    FGateWidth=np.reshape(cumFGateValsSynthOri[ii,:,:,:,:],([numberOfRealizations,-1]))
    IGateWidth=np.reshape(cumIGateValsSynthOri[ii,:,:,:,:],([numberOfRealizations,-1]))
    logRatioWidth=np.log(np.ma.masked_invalid((FGateWidth)/(IGateWidth)))
    logRatioWidthMean=np.nanmean(logRatioWidth)
    logRatioWidthSEM=scipy.stats.sem(np.nanmean(logRatioWidth,axis=1),nan_policy='omit')
    

    widthRatioMeans.append(logRatioWidthMean)
    widthRatioSEMs.append(logRatioWidthSEM)
    
    
    
logMeanObj=np.empty(len(nHiddSizeVec))
logMeanObj[:]=np.nan
MeanObj=logMeanObj
objRatioMeans=[]
objRatioSEMs=[]
# note that some I Gate values are zero and you will see the runtime warning divide by zero ...
# we are ignoring those values in here
for ii in range(len(nHiddSizeVec)):
    FGateObj=np.reshape(cumFGateValsSynthObj[ii,:,:,:,:],([numberOfRealizations,-1]))
    IGateObj=np.reshape(cumIGateValsSynthObj[ii,:,:,:,:],([numberOfRealizations,-1]))
    logRatioObj=np.log(np.ma.masked_invalid((FGateObj)/(IGateObj)))
    logRatioObjMean=np.nanmean(logRatioObj)
    logRatioObjSEM=scipy.stats.sem(np.nanmean(logRatioObj,axis=1),nan_policy='omit')
    
   
    objRatioMeans.append(logRatioObjMean)
    objRatioSEMs.append(logRatioObjSEM)



labels = ['20', '40', '80', '160', '320','640']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, objRatioMeans, width,yerr= objRatioSEMs,label='Object',color='blue',capsize=5)
rects2 = ax.bar(x + width/2, widthRatioMeans, width,yerr=widthRatioSEMs, label='Width',color='lightcoral',capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Log (Forget Gate / Input Gate)',fontsize=24)
ax.set_xlabel('Network Size',fontsize=24)
ax.set_title('Forget to input gate ratios across different network sizes',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26)
fig.tight_layout()



plt.show()



# accuracies
##################  for Synthetic Dataset
lowerLim=50
yticks=[90,100]
ylabels = [ '90', '100']
#################   for COIL-100 Dataset
lowerLim=50
yticks=[40,50,60,70,80,90,100]
ylabels = ['40','50','60','70', '80', '90', '100']
###############################################################################

widthAccSEM=np.empty(6)
widthAccSEM[:]=np.nan
for i in range(accResultLSTMFixedWidth.shape[0]):
     widthAccSEM[i]=(scipy.stats.sem(accResultLSTMFixedWidth[i,:]))

widthAccMean=np.empty(6)
for i in range(accResultLSTMFixedWidth.shape[0]):
     widthAccMean[i]=(np.mean(accResultLSTMFixedWidth[i,:]))
    
    

ObjAccSEM=np.empty(6)
ObjAccSEM[:]=np.nan

for i in range(accResultLSTMRotObj.shape[0]):
     ObjAccSEM[i]=(scipy.stats.sem(accResultLSTMRotObj[i,:]))

ObjAccMean=np.empty(6)
for i in range(accResultLSTMRotObj.shape[0]):
     ObjAccMean[i]=(np.mean(accResultLSTMRotObj[i,:]))
     
     



xlabels = ['20', '40', '80', '160', '320','640']



x = np.arange(len(xlabels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ObjAccMean*100, width,yerr= ObjAccSEM*100,label='Object',color='blue',capsize=5)
rects2 = ax.bar(x + width/2, widthAccMean*100, width,yerr=widthAccSEM*100, label='Widrh',color='lightcoral',capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Training Accuracy (%)',fontsize=24)
ax.set_xlabel('Network Size',fontsize=24)
ax.set_title('Training accuracy across different network sizes',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(xlabels,fontsize=24)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26)
ax.set_ylim(lowerLim, 105)
#fig.tight_layout()



plt.show()


















import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
nHiddSizeVec=[20,40,80,160,320,640]
numberOfRealizations=30

logMeanWidth=np.empty(len(nHiddSizeVec))
logMeanWidth[:]=np.nan
MeanWidth=logMeanWidth
widthRatioMeans=[]
widthRatioSEMs=[]

for ii in range(len(nHiddSizeVec)):

    FGateWidth=np.reshape(cumFGateValsSynthOri[ii,:,:,:,:],([numberOfRealizations,-1]))
    IGateWidth=np.reshape(cumIGateValsSynthOri[ii,:,:,:,:],([numberOfRealizations,-1]))
    logRatioWidth=(np.ma.masked_invalid((FGateWidth)/(IGateWidth)))
    logRatioWidthMean=np.nanmean(logRatioWidth)
    logRatioWidthSEM=scipy.stats.sem(np.nanmean(logRatioWidth,axis=1),nan_policy='omit')
    

    widthRatioMeans.append(logRatioWidthMean)
    widthRatioSEMs.append(logRatioWidthSEM)
    
    
    
logMeanObj=np.empty(len(nHiddSizeVec))
logMeanObj[:]=np.nan
MeanObj=logMeanObj
objRatioMeans=[]
objRatioSEMs=[]
# note that some I Gate values are zero and you will see the runtime warning divide by zero ...
# we are ignoring those values in here
for ii in range(len(nHiddSizeVec)):
    FGateObj=np.reshape(cumFGateValsSynthObj[ii,:,:,:,:],([numberOfRealizations,-1]))
    IGateObj=np.reshape(cumIGateValsSynthObj[ii,:,:,:,:],([numberOfRealizations,-1]))
    logRatioObj=(np.ma.masked_invalid((FGateObj)/(IGateObj)))
    logRatioObjMean=np.nanmean(logRatioObj)
    logRatioObjSEM=scipy.stats.sem(np.nanmean(logRatioObj,axis=1),nan_policy='omit')
    
   
    objRatioMeans.append(logRatioObjMean)
    objRatioSEMs.append(logRatioObjSEM)

logRatioObj=logRatioObj[~np.isnan(logRatioObj)]
plt.figure()
plt.bar(range(6),np.log10(objRatioMeans))




labels = ['20', '40', '80', '160', '320','640']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, objRatioMeans, width,yerr= objRatioSEMs,label='Object',color='blue',capsize=5,log=True)
rects2 = ax.bar(x + width/2, widthRatioMeans, width,yerr=widthRatioSEMs, label='Width',color='lightcoral',capsize=5,log=True)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Log (Forget Gate / Input Gate)',fontsize=24)
ax.set_xlabel('Network Size',fontsize=24)
ax.set_title('Forget to input gate ratios across different network sizes',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26)
fig.tight_layout()



plt.show()


import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
nHiddSizeVec=[20,40,80,160,320,640]
numberOfRealizations=30

logMeanWidth=np.empty(len(nHiddSizeVec))
logMeanWidth[:]=np.nan
MeanWidth=logMeanWidth
widthRatioMeans=[]
widthRatioSEMs=[]

    
logMeanObj=np.empty(len(nHiddSizeVec))
logMeanObj[:]=np.nan
MeanObj=logMeanObj
objRatioMeans=[]
objRatioSEMs=[]
# note that some I Gate values are zero and you will see the runtime warning divide by zero ...
# we are ignoring those values in here
for ii in range(len(nHiddSizeVec)):
    FGateObj=np.reshape(cumFGateValsSynthObj[ii,:,:,:,:],([numberOfRealizations,-1]))
    IGateObj=np.reshape(cumIGateValsSynthObj[ii,:,:,:,:],([numberOfRealizations,-1]))
    logRatioObj=np.log10(np.ma.masked_invalid((FGateObj)/(IGateObj)))
    logRatioObj=logRatioObj[~np.isnan(logRatioObj)]
    FGateWidth=np.reshape(cumFGateValsSynthOri[ii,:,:,:,:],([numberOfRealizations,-1]))
    IGateWidth=np.reshape(cumIGateValsSynthOri[ii,:,:,:,:],([numberOfRealizations,-1]))
    logRatioWidth=np.log10(np.ma.masked_invalid((FGateWidth)/(IGateWidth)))
    logRatioWidth=logRatioWidth[~np.isnan(logRatioWidth)]
    plt.figure()
    plt.hist(np.log10(logRatioObj),bins=400,log=True)
    plt.hist(np.log10(logRatioWidth),bins=400,log=True)
    plt.title(nHiddSizeVec[ii])
    plt.show()
    
    
   