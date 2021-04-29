# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:23:48 2020

@author: aalipour
"""
import numpy as np
from matplotlib import pyplot as plt
ObjAccMean=[.56,.87,.12]
widthAccMean=[.83,.42,.59]
xlabels = ['Synthetic','COIL-100','MIRO (Incongruent Features) ','MIRO (Congruent Features) ']


x=np.array([2,4,6,8])
obj= [48, 61,70 ,47]
ori = [48, 59,71 ,51]


ax = plt.subplot(111)
ax.bar(x-0.1, obj, width=0.2, align='center',label='Object',color='blue',capsize=5)
ax.bar(x+0.1, ori, width=0.2, align='center',label='Orientation',color='lightcoral',capsize=5)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Test Accuracy (%)',fontsize=24)
#ax.set_xlabel('Network Size',fontsize=24)
ax.set_title('Test accuracy in Feedforward Classifiers',fontsize=28)
ax.set_xticks(x)
ax.set_xticklabels(xlabels,fontsize=20)
#ax.set_yticks(yticks)
#ax.set_yticklabels(ylabels,fontsize=24)
ax.tick_params(axis="y", labelsize=24)
ax.legend(fontsize=26,loc='upper left')
ax.set_ylim(30, 80)
# fig.tight_layout()



plt.show()
