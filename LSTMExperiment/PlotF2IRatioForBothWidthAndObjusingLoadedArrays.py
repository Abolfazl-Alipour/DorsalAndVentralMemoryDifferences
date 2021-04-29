# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:31:03 2020

@author: aalipour
"""

import numpy as np
import matplotlib 
import scipy.stats 
numberOfRealizations=F2IRatioWidth.shape[1]
widthSEM=np.empty(40)
widthSEM[:]=np.nan
widthaccMean=np.empty(40)
widthaccMean[:]=np.nan

RotObjSEM=np.empty(10)
RotObjSEM[:]=np.nan
RotObjaccMean=np.empty(10)
RotObjaccMean[:]=np.nan
for i in range(F2IRatioWidth.shape[2]):
     widthSEM[i]=(scipy.stats.sem(F2IRatioWidth[0,:,i]))

widthMean=np.empty(40)
for i in range(F2IRatioWidth.shape[2]):
     widthMean[i]=(np.mean(F2IRatioWidth[0,:,i]))
     
     
for i in range(accResultsLSTMFixedWidth.shape[2]):
    widthaccMean[i]=(np.mean(accResultsLSTMFixedWidth[0,:,i]))
     

for i in range(F2IRatioRotObj.shape[2]):
     RotObjSEM[i]=(scipy.stats.sem(F2IRatioRotObj[0,:,i]))
     

RotObjhMean=np.empty(10)
for i in range(F2IRatioRotObj.shape[2]):
     RotObjhMean[i]=(np.mean(F2IRatioRotObj[0,:,i]))

for i in range(accResultsLSTMRotObj.shape[2]):
     RotObjaccMean[i]=(np.mean(accResultsLSTMRotObj[0,:,i]))




# width and obj recog
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
xTicks=np.arange(1,11)
ax1.errorbar(xTicks,RotObjhMean,yerr=RotObjSEM,fmt='s',color='b',capsize=10,ms=10)
ax2.errorbar(xTicks,widthMean[0:10],yerr=widthSEM[0:10],fmt='s',color='r',capsize=10,ms=10)
plt.yscale('log')
#plt.fill_between(range(50), widthMean-widthSTD, widthMean+widthSTD,alpha=0.5, facecolor='lightcoral')
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





# just the width
fig = plt.figure()
ax1 = fig.add_subplot(111)

xTicks=np.arange(1,41)
ax1.errorbar(xTicks,widthMean,yerr=widthSEM,fmt='s',color='r',capsize=10,ms=10)
plt.yscale('log')
#plt.fill_between(range(50), widthMean-widthSTD, widthMean+widthSTD,alpha=0.5, facecolor='lightcoral')
#plt.title('Mean Forget Gate to Input Gate Ratio ({} realizations)' .format(numberOfRealizations),fontsize=24)
#plt.xlabel('Accuracy After Each training Epoch',fontsize=32)
#plt.ylabel('Mean Forget Gate to Input Gate Ratio',fontsize=24)

ax1.set_xticks(xTicks)
ax1.set_xticklabels((np.around( widthaccMean*100,decimals=0)),fontsize=14)
#ax2.set_xticks(xTicks)
#ax2.set_xticklabels((np.around( RotObjaccMean*100,decimals=0 )),fontsize=14)

matplotlib.rc('ytick', labelsize=32) 
axes = plt.gca()
#axes.set_ylim(float(10**33),float(6*(10**33)))

plt.show()




import matplotlib.pyplot as plt

fig,(ax1,ax2) = plt.subplots(2, 1, sharex=False)

xTicks=np.arange(1,11)
xTicks2=np.arange(0.5,11.5)

ax1.errorbar(xTicks,RotObjhMean,yerr=RotObjSEM,fmt='s',color='b',capsize=10,ms=10)
ax1.set_yscale('log')
ax2.errorbar(xTicks,widthMean[30:40],yerr=widthSEM[30:40],fmt='s',color='r',capsize=10,ms=10)
ax2.set_yscale('log')

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(float(7*10**31),float(10**34)) # most of the data
ax2.set_ylim(10,10**10) # outliers only

# hide the spines between ax and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.xaxis.tick_top()
ax1.tick_params(axis='x',tick1On=False,label1On=False)#,tick2On=True,label1On=False,label2On=False) 


ax2.xaxis.tick_bottom()
ax2.tick_params(labelbottom='on') 

ax1.xaxis.tick_top()

ax1.set_xticks(xTicks)
ax1.set_xticklabels((np.around( RotObjaccMean*100,decimals=0 )),fontsize=20,color='b')
ax1.tick_params(axis='x',tick1On=False,label1On=False)#,tick2On=True,label1On=False,label2On=False) 

ax2.set_xticks(xTicks)
ax2.set_xticklabels((np.around( widthaccMean*100,decimals=0))[30:40],fontsize=20,color='r')
ax2.set_xlabel('Accuracy After Each training Epoch',fontsize=24)#,va='bottom')

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




#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#import matplotlib.pyplot as plt
#
#fig, ax1 = plt.subplots()
#
#ax1 = fig.add_subplot(111)
#
#ax1.plot(range(5), range(5))
#
#
#
#ax2 = ax1.twiny()
#ax1Xs = ax1.get_xticks()
#
#ax2Xs = []
#for X in ax1Xs:
#    ax2Xs.append(X * 2)
#
#ax2.set_xticks(ax1Xs)
#ax2.set_xbound(ax1.get_xbound())
#ax2.set_xticklabels(ax2Xs)
#
#title = ax1.set_title("Upper x-axis ticks are lower x-axis ticks doubled!")
#title.set_y(1.1)
#fig.subplots_adjust(top=0.85)
#
#fig.savefig("1.png")
#
#import matplotlib.pyplot as plt
#
#
#def make_patch_spines_invisible(ax):
#    ax.set_frame_on(True)
#    ax.patch.set_visible(False)
#    for sp in ax.spines.values():
#        sp.set_visible(False)
#
#
#fig, host = plt.subplots()
#fig.subplots_adjust(right=0.75)
#
#par1 = host.twiny()
#par2 = host.twiny()
#
## Offset the right spine of par2.  The ticks and label have already been
## placed on the right by twinx above.
#par2.spines["right"].set_position(("axes", 1.2))
## Having been created by twinx, par2 has its frame off, so the line of its
## detached spine is invisible.  First, activate the frame but make the patch
## and spines invisible.
#make_patch_spines_invisible(par2)
## Second, show the right spine.
#par2.spines["right"].set_visible(True)
#
#p1, = host.plot([0, 1, 2], [0, 1, 2], "b-", label="Density")
#p2, = par1.plot([0, 1, 2], [0, 3, 2], "r-", label="Temperature")
#p3, = par2.plot([0, 1, 2], [50, 30, 15], "g-", label="Velocity")
#
#host.set_xlim(0, 2)
#host.set_ylim(0, 2)
#par1.set_ylim(0, 4)
#par2.set_ylim(1, 65)
#
#host.set_xlabel("Distance")
#host.set_ylabel("Density")
#par1.set_ylabel("Temperature")
#par2.set_ylabel("Velocity")
#
#host.yaxis.label.set_color(p1.get_color())
#par1.yaxis.label.set_color(p2.get_color())
#par2.yaxis.label.set_color(p3.get_color())
#
#tkw = dict(size=4, width=1.5)
#host.tick_params(axis='y', colors=p1.get_color(), **tkw)
#par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
#par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
#host.tick_params(axis='x', **tkw)
#
#lines = [p1, p2, p3]
#
#host.legend(lines, [l.get_label() for l in lines])
#
#plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#import numpy as np
#import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d.axes3d as axes3d
#
#fig = plt.figure(dpi=100)
#ax = fig.add_subplot(111, projection='3d')
#
#
#
##data
#fx = range(10)
#fy = RotObjhMean
#fz = RotObjaccMean
#
##error data
##xerror = [0.041504064,0.02402152,0.059383144]
#yerror = RotObjSEM
##zerror = [3.677693713,1.345712547,0.724095592]
#
##plot points
#ax.plot(fx, fy, fz, linestyle="None", marker="o")
#
##plot errorbars
#for i in np.arange(0, len(fx)):
#    ax.plot([fx[i], fx[i]], [fy[i], fy[i]], [fz[i], fz[i]], marker="_")
#    ax.plot([fx[i], fx[i]], [fy[i]+yerror[i], fy[i]-yerror[i]], [fz[i], fz[i]], marker="_")
#    ax.plot([fx[i], fx[i]], [fy[i], fy[i]], [fz[i], fz[i]], marker="_")
#
##configure axes
##ax.set_xlim3d(0.55, 0.8)
##ax.set_ylim3d(0.2, 0.5)
##ax.set_zlim3d(8, 19)
#
#plt.show()