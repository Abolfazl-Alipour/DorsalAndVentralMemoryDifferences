#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:30:19 2020

@author: aalipour
"""




import torch
import pickle
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from loadPretrainedclassifierAndIncorporateitintoESNForRotatingObjLeakyRatechangesMIROFinegrain import ESNRotatingObjectLrnLeakyRate
from loadPretrainedclassifierAndIncorporateitintoESNForFixedWidthLeakyRatechangesMIROFinegrain import ESNFixedWidthLrnLeakyRate
from tqdm import tqdm
import pdb
import seaborn as sns
import pandas as pd
numberOfRealizations=30
# intialLeakyRateVec=[.25,.5,.75]
intialLeakyRateVec=np.random.rand(1,numberOfRealizations)
nIterations=18#22



rotatObjleakyRateMat=torch.empty(numberOfRealizations,len(intialLeakyRateVec),nIterations)
rotatObjleakyRateMat[:]=np.nan

fixedWidthleakyRateMat=torch.empty(numberOfRealizations,len(intialLeakyRateVec),nIterations)
fixedWidthleakyRateMat[:]=np.nan



rotatObjAccuracyMat=torch.empty(numberOfRealizations,len(intialLeakyRateVec),nIterations)
rotatObjAccuracyMat[:]=np.nan

fixedWidthAccuracyMat=torch.empty(numberOfRealizations,len(intialLeakyRateVec),nIterations)
fixedWidthAccuracyMat[:]=np.nan



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

# for j in tqdm(range(len(randomSeedList))):
#     i=0
#     for leaky_rate in intialLeakyRateVec:
#         tmp=ESNRotatingObjectLrnLeakyRate(leaky_rate,nIterations,randomSeedList[j],n_hidden = 160)
#         rotatObjAccuracyMat[j,i,0:nIterations]=torch.FloatTensor(tmp[0])
#         rotatObjleakyRateMat[j,i,0:nIterations]=torch.FloatTensor(tmp[1]).t()
#         i=i+1






# # Saving the results:
# with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNObjLeakyRateLrnsMIROMidSizeFinegrainNtwSze160-8ClassExp.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([rotatObjAccuracyMat, rotatObjleakyRateMat], f)
#     f.close()
# # # # Getting back the objects:
# with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNObjLeakyRateLrnsMIROMidSizeFinegrainNtwSze160-8ClassExp.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     rotatObjAccuracyMat, rotatObjleakyRateMat = pickle.load(f)







# for j in tqdm(range(len(randomSeedList))):
#     i=0
#     for leaky_rate in intialLeakyRateVec:
#         tmp=ESNFixedWidthLrnLeakyRate(leaky_rate,nIterations,randomSeedList[j],numOfClasses=8,n_hidden = 160)
# #        pdb.set_trace()
#         fixedWidthAccuracyMat[j,i,0:nIterations]=torch.FloatTensor(tmp[0])
#         fixedWidthleakyRateMat[j,i,0:nIterations]=torch.FloatTensor(tmp[1]).t()
#         i=i+1


# # Saving the results:
# with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNWidthLeakyRateLrnsMIROMidSizeFinegrainNtwSze160-8ClassExp.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([fixedWidthAccuracyMat, fixedWidthleakyRateMat], f)
#     f.close()
# # # # Getting back the objects:
# with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNWidthLeakyRateLrnsMIROMidSizeFinegrainNtwSze160.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     fixedWidthAccuracyMat, fixedWidthleakyRateMat = pickle.load(f)


for j in tqdm(range(len(randomSeedList))):
        i=0

        tmp=ESNRotatingObjectLrnLeakyRate(intialLeakyRateVec[0,j],nIterations,randomSeedList[j],n_hidden = 160)
        rotatObjAccuracyMat[j,i,0:nIterations]=torch.FloatTensor(tmp[0])
        rotatObjleakyRateMat[j,i,0:nIterations]=torch.FloatTensor(tmp[1]).t()
        i=i+1






# Saving the results:
with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNObjLeakyRateLrnsMIROMidSizeFinegrainNtwSze160-8ClassExpRandLeak.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([rotatObjAccuracyMat, rotatObjleakyRateMat], f)
    f.close()
# # # Getting back the objects:
with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNObjLeakyRateLrnsMIROMidSizeFinegrainNtwSze160-8ClassExpRandLeak.pkl','rb') as f:  # Python 3: open(..., 'rb')
    rotatObjAccuracyMat, rotatObjleakyRateMat = pickle.load(f)







for j in tqdm(range(len(randomSeedList))):
        i=0
    
        tmp=ESNFixedWidthLrnLeakyRate(intialLeakyRateVec[0,j],nIterations,randomSeedList[j],numOfClasses=8,n_hidden = 160)
#        pdb.set_trace()
        fixedWidthAccuracyMat[j,i,0:nIterations]=torch.FloatTensor(tmp[0])
        fixedWidthleakyRateMat[j,i,0:nIterations]=torch.FloatTensor(tmp[1]).t()
        i=i+1


# Saving the results:
with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNWidthLeakyRateLrnsMIROMidSizeFinegrainNtwSze160-8ClassExpRandLeak.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([fixedWidthAccuracyMat, fixedWidthleakyRateMat], f)
    f.close()
# # # Getting back the objects:
with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNWidthLeakyRateLrnsMIROMidSizeFinegrainNtwSze160-8ClassExpRandLeak.pkl','rb') as f:  # Python 3: open(..., 'rb')
    fixedWidthAccuracyMat, fixedWidthleakyRateMat = pickle.load(f)



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

#############################################################################

fixedWidthleakyRateMat=fixedWidthleakyRateMat.numpy()
fixedWidthAccuracyMat=fixedWidthAccuracyMat.numpy()
rotatObjAccuracyMat=rotatObjAccuracyMat.numpy()
rotatObjleakyRateMat=rotatObjleakyRateMat.numpy()

        
import numpy as np
objSEM=np.empty([3,nIterations])
obj=rotatObjleakyRateMat
nanValuesObj=np.argwhere(np.isnan(obj))
if nanValuesObj.size!=0:
    obj=np.delete(obj,nanValuesObj[0,0],axis=0)
for i in range(obj.shape[1]):
    for j in range(obj.shape[2]):
        objSEM[i,j]=scipy.stats.sem(obj[:,i,j])

objMean=np.empty([3,nIterations])
for i in range(obj.shape[1]):
    for j in range(obj.shape[2]):
        objMean[i,j]=(np.mean(obj[:,i,j]))
     

       
import numpy as np
objAccSEM=np.empty([3,nIterations])
objAcc=rotatObjAccuracyMat
for i in range(objAcc.shape[1]):
    for j in range(objAcc.shape[2]):
        objAccSEM[i,j]=scipy.stats.sem(objAcc[:,i,j])

objAccMean=np.empty([3,nIterations])
for i in range(objAcc.shape[1]):
    for j in range(objAcc.shape[2]):
        objAccMean[i,j]=(np.mean(objAcc[:,i,j]))

        
import numpy as np
widthSEM=np.empty([3,nIterations])
widthSEM[:]=np.nan
width=fixedWidthleakyRateMat
nanValuesWidth=np.argwhere(np.isnan(width))
if nanValuesWidth.size!=0:
    width=np.delete(obj,nanValuesWidth[0,0],axis=0)
    
for i in range(width.shape[1]):
    for j in range(width.shape[2]):
      widthSEM[i,j]=(scipy.stats.sem(width[:,i,j]))

widthMean=np.empty([3,nIterations])
widthMean[:]=np.nan
for mm in range(width.shape[1]):
    for kk in range(width.shape[2]):
        widthMean[mm,kk]=(np.mean(width[:,mm,kk]))

       

widthAccSEM=np.empty([3,nIterations])
widthAccSEM[:]=np.nan
widthAcc=fixedWidthAccuracyMat

    
for i in range(widthAcc.shape[1]):
    for j in range(widthAcc.shape[2]):
      widthAccSEM[i,j]=(scipy.stats.sem(widthAcc[:,i,j]))

widthAccMean=np.empty([3,nIterations])
widthAccMean[:]=np.nan
for mm in range(widthAcc.shape[1]):
    for kk in range(widthAcc.shape[2]):
        widthAccMean[mm,kk]=(np.mean(widthAcc[:,mm,kk]))   
     

# scatterplot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Create data
g25obj = (objAccMean[0,:]*100,objMean[0,:])
g50obj = (objAccMean[1,:]*100,objMean[1,:])
g75obj = (objAccMean[2,:]*100,objMean[2,:])
g25width = (widthAccMean[0,:]*100,widthMean[0,:])
g50width = (widthAccMean[1,:]*100,widthMean[1,:])
g75width = (widthAccMean[2,:]*100,widthMean[2,:])
data = (g25obj, g50obj, g75obj,g25width,g50width,g75width)
colorsObj = ("cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","royalblue","royalblue","royalblue","royalblue","b","b","b","mediumblue","mediumblue","darkblue","k")
colorsWidth=("lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","indianred","indianred","indianred","indianred","brown","brown","brown","firebrick","firebrick","maroon","k")




# Create plot
sns.set_theme( )
# sns.axes_style("whitegrid")
markerSize=150
fig = plt.figure()
#plt.title('Leaky Rate VS. Classification Accuracy')

ax = fig.add_subplot(1, 1, 1)
ax.tick_params(labelsize=22) 
ax.scatter(g25obj[0], 1-g25obj[1], alpha=0.8, c=colorsObj, edgecolors='none', s=markerSize,label='Object classification Mem. Coef.')
ax.scatter(g25width[0], 1-g25width[1], alpha=0.8, c=colorsWidth, edgecolors='none', s=markerSize,label='Orientation classification Mem. Coef.')

# ax.scatter(0.2,.75, alpha=0.8, marker='>',c='k', edgecolors='none', s=300,label='Initial Mem. Coef.')
ax.legend(fontsize=30)

ax.set_xlabel('Network Accuracy (%)',size=24)
ax.set_ylabel('Memory Coefficient',size=24)

ax.set_xlim(0, 110)
ax.set_ylim(0.2, .8)

#for 0.5
ax = fig.add_subplot(1, 3, 2)
ax.tick_params(labelsize=22)
# ax.set_yticklabels(labels='')
ax.scatter(g50obj[0], 1-g50obj[1], alpha=0.8, c=colorsObj, edgecolors='none', s=markerSize,label='Object classification Mem. Coef.')
ax.scatter(g50width[0], 1-g50width[1], alpha=0.8, c=colorsWidth, edgecolors='none', s=markerSize,label='Orientation classification Mem. Coef.')
ax.set_xlim(0, 100)
ax.set_ylim(0.15, 1)
ax.scatter(0.2,.5, alpha=0.8, marker='>',c='k', edgecolors='none', s=600,label='Initial Mem. Coef.')
ax.set_xlabel('Network Accuracy (%)',size=24)
#ax.legend(fontsize=14)
# for 0.75
ax = fig.add_subplot(1, 3, 3)


# ax.set_yticklabels(labels='') 
ax.scatter(0.2,.25, alpha=0.8, marker='>',c='k', edgecolors='none', s=600,label='Initial Memory Coefficient')
ax.scatter(g75obj[0], 1-g75obj[1], alpha=0.8, c=colorsObj, edgecolors='none', s=markerSize,label='Object classification Mem. Coef.')    
ax.scatter(g75width[0], 1-g75width[1], alpha=0.8, c=colorsWidth, edgecolors='none', s=markerSize,label='Orientation classification Mem. Coef.')
ax.set_xlim(0, 100)
ax.set_ylim(0.15, 1)
ax.tick_params(labelsize=22)

# ax.plot(range(100),(np.ones(100)*(1-0.75)), 'k--', )
ax.set_xlabel('Network Accuracy (%)',size=24)
#ax.legend(fontsize=14,loc=7)


plt.show()


# ###############################



################################





fig = plt.figure()
plt.rcParams["axes.labelsize"] = 26
# plt.title('Forget Rate VS. Classification Accuracy after 18 Epochs')
sns.axes_style("darkgrid")
# ax=sns.boxplot( data=[1-obj[:,2,17],1-width[:,2,17]])
plt.show()                
capSizeVal=0.05
ax1 = fig.add_subplot(1, 3, 1)
df1 = pd.melt( pd.DataFrame( {"Object":1-obj[:,0,17], "Orientation":1-width[:,0,17]}), var_name = 'Task', value_name = 'Memory Coefficient') 
myPal=sns.color_palette(["b","r"])
sns.pointplot(x='Task', y='Memory Coefficient', data=df1,capsize=capSizeVal,palette=myPal,ci= 'sd',join=False)
# sns.boxplot( data=[1-obj[:,0,17],1-width[:,0,17]],ax=ax1,palette=myPal).set(ylabel='Memory Coefficient', xticklabels=['Object','Orientation'])
ax1.set_ylim(.35, 1)
ax1.tick_params(labelsize=22) 
ax1.set_xlabel("")


# ax.set_xlabel('Network Accuracy (%)',size=24)

# ax.set_xlim(75, 100)
# ax.set_ylim(0.1, 0.6)

#for 0.5
ax2 = fig.add_subplot(1, 3, 2)
ax2.tick_params(labelsize=16) 
ax2.set_ylabel('Memory Coefficient',size=24)
df2 = pd.melt( pd.DataFrame( {"Object":1-obj[:,1,17], "Orientation":1-width[:,1,17]}), var_name = 'Task', value_name = 'Memory Coefficient') 

sns.pointplot(x='Task', y='Memory Coefficient', data=df2, capsize=capSizeVal,palette=myPal,ci= 'sd',join=False).set(yticklabels=[],xticklabels=['Object','Orientation'])
# sns.boxplot( data=[1-obj[:,1,17],1-width[:,1,17]],ax=ax2, palette=myPal).set(yticklabels=[],xticklabels=['Object','Orientation'])
ax2.set_ylim(.35, 1)
ax2.tick_params(labelsize=22) 
ax2.set_ylabel("")
ax2.set_xlabel("")



ax3 = fig.add_subplot(1, 3, 3)

df3= pd.melt( pd.DataFrame( {"Object":1-obj[:,2,17], "Orientation":1-width[:,2,17]}), var_name = 'Task', value_name = 'Memory Coefficient') 

sns.pointplot(x='Task', y='Memory Coefficient', data=df3,capsize=capSizeVal,palette=myPal,ci= 'sd',join=False).set(yticklabels=[],xticklabels=['Object','Orientation'])
# sns.boxplot( data=[1-obj[:,2,17],1-width[:,2,17]],ax=ax3,palette=myPal).set(yticklabels=[],xticklabels=['Object','Orientation'])
ax3.set_ylim(.35, 1)
ax3.set_ylabel("")
ax3.tick_params(labelsize=22) 
ax3.set_xlabel("")





# Create plot
sns.set_theme()
# sns.axes_style("whitegrid")
fig = plt.figure()
#plt.title('Leaky Rate VS. Classification Accuracy')

ax = fig.add_subplot(1, 3, 1)
ax.tick_params(labelsize=16) 
ax.scatter(g25obj[0], 1-g25obj[1], alpha=0.8, c=colorsObj, edgecolors='none', s=30,label='Object classification Memory Coefficient')
ax.scatter(g25width[0], 1-g25width[1], alpha=0.8, c=colorsWidth, edgecolors='none', s=30,label='Orientation classification Memory Coefficient')

ax.scatter(0.2,.75, alpha=0.8, marker='>',c='k', edgecolors='none', s=200,label='Initial Memory Coefficient')
ax.legend(fontsize=11)

ax.set_xlabel('Network Accuracy (%)',size=24)
ax.set_ylabel('Memory Coefficient',size=24)

ax.set_xlim(0, 100)
ax.set_ylim(0.20, 1)

#for 0.5
ax = fig.add_subplot(1, 3, 2)
ax.tick_params(labelsize=16) 
ax.scatter(g50obj[0], 1-g50obj[1], alpha=0.8, c=colorsObj, edgecolors='none', s=30,label='Object classification Memory Coefficient')
ax.scatter(g50width[0], 1-g50width[1], alpha=0.8, c=colorsWidth, edgecolors='none', s=30,label='Orientation classification Memory Coefficient')
ax.set_xlim(0, 100)
ax.set_ylim(0.20, 1)
xy_line = (0.5, 0.5)
ax.scatter(0.2,.5, alpha=0.8, marker='>',c='k', edgecolors='none', s=300,label='Initial Memory Coefficient')
ax.set_xlabel('Network Accuracy (%)',size=24)
#ax.legend(fontsize=14)
# for 0.75
ax = fig.add_subplot(1, 3, 3)
ax.tick_params(labelsize=16) 
ax.scatter(0.2,.25, alpha=0.8, marker='>',c='k', edgecolors='none', s=300,label='Initial Memory Coefficient')
ax.scatter(g75obj[0], 1-g75obj[1], alpha=0.8, c=colorsObj, edgecolors='none', s=30,label='Object classification Memory Coefficient')    
ax.scatter(g75width[0], 1-g75width[1], alpha=0.8, c=colorsWidth, edgecolors='none', s=30,label='Orientation classification Memory Coefficient')
ax.set_xlim(0, 100)
ax.set_ylim(0.2, 1)
ax.yticklabels=[]

# ax.plot(range(100),(np.ones(100)*(1-0.75)), 'k--', )
ax.set_xlabel('Network Accuracy (%)',size=24)
#ax.legend(fontsize=14,loc=7)


plt.show()


###############################


# Create plot for final points
fig = plt.figure()
# plt.title('Memory Coefficient VS. Classification Accuracy after 18 Epochs')
capSizeVal=2
ax = fig.add_subplot(3, 1, 1)
ax.tick_params(labelsize=16) 
ax.errorbar(g25obj[0][-1], 1-g25obj[1][-1], yerr=objSEM[0,-1],xerr=objAccSEM[0,-1]*100,capsize=capSizeVal,label='Object classification Memory Coefficient')
ax.errorbar(g25width[0][-1], 1-g25width[1][-1], yerr=widthSEM[0,-1],xerr=widthAccSEM[0,-1]*100,capsize=capSizeVal,label='Orientation classification Memory Coefficient')

ax.plot(range(100),(np.ones(100)*(1-0.25)), 'k--', label='Initial Memory Coefficient')
# ax.set_xlabel('Network Accuracy (%)',size=24)

ax.legend(fontsize=12)
ax.set_xlim(40, 90)
# ax.set_ylim(0.1, 0.6)

#for 0.5
ax = fig.add_subplot(3, 1, 2)
ax.tick_params(labelsize=16) 
ax.errorbar(g50obj[0][-1], 1-g50obj[1][-1], yerr=objAccSEM[1,-1],xerr=objAccSEM[1,-1]*100,capsize=capSizeVal,label='Object classification Memory Coefficient')
ax.errorbar(g50width[0][-1], 1-g50width[1][-1], yerr=widthSEM[1,-1],xerr=widthAccSEM[1,-1]*100,capsize=capSizeVal,label='Width classification Memory Coefficient')
ax.set_xlim(40, 90)
# ax.set_ylim(0.1, 0.8)

ax.plot(range(100),(np.ones(100)*0.5), 'k--', label='Initial Memory Coefficient')
# ax.set_xlabel('Network Accuracy (%)',size=24)
ax.set_ylabel('Memory Coefficient',size=24)
#ax.legend(fontsize=14)
# for 0.75
ax = fig.add_subplot(3, 1, 3)
ax.tick_params(labelsize=16) 
ax.errorbar(g75obj[0][-1], 1-g75obj[1][-1],yerr=objAccSEM[2,-1],xerr=objAccSEM[2,-1]*100,capsize=capSizeVal,label='Object classification Memory Coefficient')    
ax.errorbar(g75width[0][-1], 1-g75width[1][-1], yerr=widthSEM[2,-1],xerr=widthAccSEM[2,-1]*100,capsize=capSizeVal,label='Width classification Memory Coefficient')
ax.set_xlim(40, 90)
# ax.set_ylim(0.1, 0.8)

ax.plot(range(100),(np.ones(100)*(1-0.75)), 'k--', label='Initial Memory Coefficient')
ax.set_xlabel('Network Accuracy (%)',size=24)
#ax.legend(fontsize=14,loc=7)


plt.show()



################################################################################

fig = plt.figure()
plt.rcParams["axes.labelsize"] = 26
# plt.title('Forget Rate VS. Classification Accuracy after 18 Epochs')
sns.axes_style("darkgrid")
# ax=sns.boxplot( data=[1-obj[:,2,17],1-width[:,2,17]])
plt.show()                
capSizeVal=2
ax1 = fig.add_subplot(1, 3, 1)

sns.boxplot( data=[1-obj[:,0,17],1-width[:,0,17]],ax=ax1).set(ylabel='Memory Coefficient', xticklabels=['Object','Orientation'])
ax1.set_ylim(.35, 1)
ax1.tick_params(labelsize=20) 


# ax.set_xlabel('Network Accuracy (%)',size=24)

# ax.set_xlim(75, 100)
# ax.set_ylim(0.1, 0.6)

#for 0.5
ax2 = fig.add_subplot(1, 3, 2)
ax.tick_params(labelsize=16) 
ax.set_ylabel('Memory Coefficient',size=24)
sns.boxplot( data=[1-obj[:,1,17],1-width[:,1,17]],ax=ax2).set(yticklabels=[],xticklabels=['Object','Orientation'])
ax2.set_ylim(.35, 1)
ax2.tick_params(labelsize=16) 

ax3 = fig.add_subplot(1, 3, 3)

ax.set_ylabel('Memory Coefficient',size=24)
sns.boxplot( data=[1-obj[:,2,17],1-width[:,2,17]],ax=ax3).set(yticklabels=[],xticklabels=['Object','Orientation'])
ax3.set_ylim(.35, 1)
ax3.tick_params(labelsize=16) 


# # create some data
# xy = np.random.rand(4, 2)
# xy_line = (0, 1)

# # set up figure and ax
# fig, ax = plt.subplots(figsize=(8,8))

# # create the scatter plots
# ax.scatter(xy[:, 0], xy[:, 1], c='blue')
# for point, name in zip(xy, 'ABCD'):
#     ax.annotate(name, xy=point, xytext=(0, -10), textcoords='offset points',
#                 color='blue', ha='center', va='center')
# ax.scatter([0], [1], c='black', s=60)
# ax.annotate('Perfect Classification', xy=(0, 1), xytext=(0.1, 0.9),
#             arrowprops=dict(arrowstyle='->'))

# # create the line
# ax.plot(xy_line, 'r--', label='Random guess')
# ax.annotate('Better', xy=(0.3, 0.3), xytext=(0.2, 0.4),
#             arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
# ax.annotate('Worse', xy=(0.3, 0.3), xytext=(0.4, 0.2),
#             arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
# # add labels, legend and make it nicer
# ax.set_xlabel('FPR or (1 - specificity)')
# ax.set_ylabel('TPR or sensitivity')
# ax.set_title('ROC Space')
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.legend()
# plt.tight_layout()
# plt.savefig('scatter_line.png', dpi=80)


    
# #import matplotlib.pyplot as plt
# #
# #plt.figure()
# #plt.errorbar(range(50),objMean,yerr=objSTD)
# #plt.title('Network Performance- Object Classification (60 realizations)')
# #plt.xlabel('leaky rate')
# #plt.ylabel('Ratio Of correct responses to Number Of Images')
# #plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
# #plt.show()

