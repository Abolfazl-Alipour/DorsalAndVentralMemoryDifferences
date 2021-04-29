# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:33:34 2020

@author: aalipour
"""







#if plotting synthetic dataset=
ax1Ylim=float(2*10**32),float(10**34)

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

widthSEM=np.empty(6)
widthSEM[:]=np.nan
width=F2IRatioWidth
RotObjSEM=np.empty(6)
RotObjSEM[:]=np.nan
RotObj=F2IRatioRotObj
for i in range(width.shape[0]):
     widthSEM[i]=(scipy.stats.sem(width[i,:]))

widthMean=np.empty(6)
for i in range(width.shape[0]):
     widthMean[i]=(np.mean(width[i,:]))
     

for i in range(RotObj.shape[0]):
     RotObjSEM[i]=(scipy.stats.sem(RotObj[i,:]))

RotObjhMean=np.empty(6)
for i in range(RotObj.shape[0]):
     RotObjhMean[i]=(np.mean(RotObj[i,:]))

fig,(ax1,ax2) = plt.subplots(2, 1, sharex=True)

xTicks=np.arange(0,6)
xTicks2=np.arange(0.5,11.5)

ax1.errorbar( np.arange(6),RotObjhMean,yerr=RotObjSEM,fmt='s',color='b',capsize=10,ms=10)
ax1.set_yscale('log')
ax2.errorbar(np.arange(6),widthMean ,yerr=widthSEM ,fmt='>',color='lightcoral',capsize=10,ms=10)
ax2.set_yscale('log')

ax1.set_ylim(ax1Ylim) 
ax2.set_ylim(10,10**8)
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

#ax1.xaxis.tick_top()
ax1.tick_params(axis='x',tick1On=False,label1On=False)

ax2.xaxis.tick_bottom()
ax2.tick_params(labelbottom='on') 
#ax1.set_yticks(xTicks)
ax1.tick_params(axis='x',tick1On=False,label1On=False)#,tick2On=True,label1On=False,label2On=False) 

ax2.set_xticks(xTicks)

ax2.set_xticklabels([20, 40, 80, 160,320,640],fontsize=24)
plt.subplots_adjust(wspace=0.15)
matplotlib.rc('ytick', labelsize=32) 
d = .015 # how big to make the diagonal lines in axes coordinates


# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d,1+d),(-d,+d), **kwargs) # top-left diagonal
ax1.plot((-d,+d),(-d,+d), **kwargs) # bottom-left diagonal

kwargs.update(transform=ax2.transAxes) # switch to the bottom axes
ax2.plot((1-d,1+d),(1-d,1+d), **kwargs) # top-right diagonal
ax2.plot((-d,d),(1-d,1+d), **kwargs) # bottom-right diagonal
ax2.set_xlabel('Network Size',fontsize=32)#,va='bottom')

ax1.set_title('Mean Forget Gate to Input Gate Ratio (30 Realizations)',fontsize=24)
fig.text(0.06, 0.5, 'Mean Forget Gate to Input Gate Ratio',fontsize=24, ha='center', va='center', rotation='vertical')
plt.show()















#if plotting coil100:
ax1Ylim=float(10**35),float(2*10**35)

import matplotlib 
import scipy  
widthSEM=np.empty(6)
widthSEM[:]=np.nan
width=F2IRatioWidth
RotObjSEM=np.empty(6)
RotObjSEM[:]=np.nan
RotObj=F2IRatioRotObj
for i in range(width.shape[0]):
     widthSEM[i]=(scipy.stats.sem(width[i,:],nan_policy='omit'))

widthMean=np.empty(6)
for i in range(width.shape[0]):
     widthMean[i]=(np.nanmean(width[i,:]))
     

for i in range(RotObj.shape[0]):
     RotObjSEM[i]=(scipy.stats.sem(RotObj[i,:],nan_policy='omit'))

RotObjhMean=np.empty(6)
for i in range(RotObj.shape[0]):
     RotObjhMean[i]=(np.nanmean(RotObj[i,:]))





fig,(ax1,ax2) = plt.subplots(2, 1, sharex=True)

xTicks=np.arange(0,6)
xTicks2=np.arange(0.5,11.5)

ax1.errorbar( np.arange(6),RotObjhMean,yerr=RotObjSEM,fmt='s',color='b',capsize=10,ms=10)
ax1.set_yscale('log')
ax2.errorbar(np.arange(6),widthMean ,yerr=widthSEM ,fmt='>',color='lightcoral',capsize=10,ms=10)
ax2.set_yscale('log')

ax1.set_ylim(ax1Ylim) 
ax2.set_ylim(10,10**10)
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

#ax1.xaxis.tick_top()
ax1.tick_params(axis='x',tick1On=False,label1On=False)

ax2.xaxis.tick_bottom()
ax2.tick_params(labelbottom=True) 
#
#ax1.set_yticks([10**35,2*10**35])
#ax1.set_yticklabels( [ '$10^{35}$', ' $2*10^{35}$'],fontsize=18)

ax1.tick_params(axis='x',tick1On=False,label1On=False)#,tick2On=True,label1On=False,label2On=False) 

ax2.set_xticks(xTicks)

ax2.set_xticklabels([20, 40, 80, 160,320,640],fontsize=24)
plt.subplots_adjust(wspace=0.15)
matplotlib.rc('ytick', labelsize=32) 
d = .015 # how big to make the diagonal lines in axes coordinates


# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d,1+d),(-d,+d), **kwargs) # top-left diagonal
ax1.plot((-d,+d),(-d,+d), **kwargs) # bottom-left diagonal

kwargs.update(transform=ax2.transAxes) # switch to the bottom axes
ax2.plot((1-d,1+d),(1-d,1+d), **kwargs) # top-right diagonal
ax2.plot((-d,d),(1-d,1+d), **kwargs) # bottom-right diagonal
ax2.set_xlabel('Network Size',fontsize=32)#,va='bottom')

ax1.set_title('Mean Forget Gate to Input Gate Ratio (30 Realizations)',fontsize=24)
fig.text(0.06, 0.5, 'Mean Forget Gate to Input Gate Ratio',fontsize=24, ha='center', va='center', rotation='vertical')
plt.show()





# zoom-in / limit the view to different portions of the data
 # outliers only

# hide the spines between ax and ax2

plt.show()




# If you're not familiar with np.r_, don't worry too much about this. It's just 
# a series with points from 0 to 1 spaced at 0.1, and 9 to 10 with the same spacing.
x = np.r_[0:1:0.1, 9:10:0.1]
y = np.sin(x)



# plot the same data on both axes
ax.plot(x, y, 'bo')
ax2.plot(x, y, 'bo')

# zoom-in / limit the view to different portions of the data
ax.set_xlim(0,1) # most of the data
ax2.set_xlim(9,10) # outliers only

# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labeltop='off') # don't put tick labels at the top
ax2.yaxis.tick_right()

# Make the spacing between the two axes a bit smaller
plt.subplots_adjust(wspace=0.15)

plt.show()





#both in the same plot with all netwo sizes



import matplotlib 
import scipy  
widthSEM=np.empty(6)
widthSEM[:]=np.nan
width=F2IRatioWidth
RotObjSEM=np.empty(6)
RotObjSEM[:]=np.nan
RotObj=F2IRatioRotObj
for i in range(width.shape[0]):
     widthSEM[i]=(scipy.stats.sem(width[i,:]))

widthMean=np.empty(6)
for i in range(width.shape[0]):
     widthMean[i]=(np.mean(width[i,:]))
     

for i in range(RotObj.shape[0]):
     RotObjSEM[i]=(scipy.stats.sem(RotObj[i,:]))

RotObjhMean=np.empty(6)
for i in range(RotObj.shape[0]):
     RotObjhMean[i]=(np.mean(RotObj[i,:]))

plt.figure()
plt.errorbar( np.arange(6),widthMean,yerr=widthSEM,fmt='>',color='lightcoral',capsize=10,ms=10)
plt.errorbar(np.arange(6),RotObjhMean,yerr=RotObjSEM,fmt='s',color='b',capsize=10,ms=10)
plt.title('Mean Forget Gate to Input Gate Ratio ({} realizations)' .format(numberOfRealizations),fontsize=24)
plt.xlabel('Network Size',fontsize=32)
plt.ylabel('Mean Forget Gate to Input Gate Ratio',fontsize=24)
plt.xticks(ticks=[0,1,2,3,4,5 ],labels=[20, 40, 80, 160,320,640],fontsize=24)
matplotlib.rc('ytick', labelsize=32)
plt.yscale('log')
plt.show()





























#scratchpad

RotObjhMean=objRatioMeans
RotObjSEM=objRatioSEMs
widthMean=widthRatioMeans
widthSEM=widthRatioSEMs




import matplotlib.pyplot as plt
import matplotlib
import scipy.stats
import numpy as np


widthSEM=np.empty(6)
widthSEM[:]=np.nan
width=F2IRatioWidth
RotObjSEM=np.empty(6)
RotObjSEM[:]=np.nan
RotObj=F2IRatioRotObj
for i in range(width.shape[0]):
     widthSEM[i]=(scipy.stats.sem(width[i,:]))

widthMean=np.empty(6)
for i in range(width.shape[0]):
     widthMean[i]=(np.mean(width[i,:]))
     

for i in range(RotObj.shape[0]):
     RotObjSEM[i]=(scipy.stats.sem(RotObj[i,:]))

RotObjhMean=np.empty(6)
for i in range(RotObj.shape[0]):
     RotObjhMean[i]=(np.mean(RotObj[i,:]))





ax1Ylim=float(2*10**30),float(10**36)

fig,(ax1,ax2) = plt.subplots(2, 1, sharex=True)

xTicks=np.arange(0,6)
xTicks2=np.arange(0.5,11.5)

ax1.errorbar( np.arange(6),RotObjhMean,yerr=RotObjSEM,fmt='s',color='b',capsize=10,ms=10)
ax1.set_yscale('log')
ax2.errorbar(np.arange(6),widthMean ,yerr=widthSEM ,fmt='>',color='lightcoral',capsize=10,ms=10)
ax2.set_yscale('log')

ax1.set_ylim(ax1Ylim) 
ax2.set_ylim(float(10**10),float(10**23))
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

#ax1.xaxis.tick_top()
ax1.tick_params(axis='x',tick1On=False,label1On=False)

ax2.xaxis.tick_bottom()
ax2.tick_params(labelbottom='on') 
#ax1.set_yticks(xTicks)
ax1.tick_params(axis='x',tick1On=False,label1On=False)#,tick2On=True,label1On=False,label2On=False) 

ax2.set_xticks(xTicks)

ax2.set_xticklabels([20, 40, 80, 160,320,640],fontsize=24)
plt.subplots_adjust(wspace=0.15)
matplotlib.rc('ytick', labelsize=32) 
d = .015 # how big to make the diagonal lines in axes coordinates


# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d,1+d),(-d,+d), **kwargs) # top-left diagonal
ax1.plot((-d,+d),(-d,+d), **kwargs) # bottom-left diagonal

kwargs.update(transform=ax2.transAxes) # switch to the bottom axes
ax2.plot((1-d,1+d),(1-d,1+d), **kwargs) # top-right diagonal
ax2.plot((-d,d),(1-d,1+d), **kwargs) # bottom-right diagonal
ax2.set_xlabel('Network Size',fontsize=32)#,va='bottom')

ax1.set_title('Mean Forget Gate to Input Gate Ratio (30 Realizations)',fontsize=24)
fig.text(0.06, 0.5, 'Mean Forget Gate to Input Gate Ratio',fontsize=24, ha='center', va='center', rotation='vertical')
plt.show()
