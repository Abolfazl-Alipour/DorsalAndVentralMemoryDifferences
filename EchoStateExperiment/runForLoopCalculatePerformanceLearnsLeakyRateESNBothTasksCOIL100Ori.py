# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon May  4 16:33:25 2020

# @author: aalipour
# """


# # -*- coding: utf-8 -*-
# """
# Created on Tue Dec 10 14:27:07 2019

# @author: aalipour
# """





import torch
import pickle
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from loadPretrainedclassifierAndIncorporateitintoESNForRotatingObjLeakyRatechangesCOIL100 import ESNRotatingObjectLrnLeakyRate
from loadPretrainedclassifierAndIncorporateitintoESNForFixedWidthLeakyRatechangesCOIL100 import ESNFixedWidthLrnLeakyRate
from tqdm import tqdm
import pdb
numberOfRealizations=30
intialLeakyRateVec=[.25,.5,.75]
nIterations=18



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

#



for j in tqdm(range(len(randomSeedList))):
    i=0
    for leaky_rate in intialLeakyRateVec:
        tmp=ESNFixedWidthLrnLeakyRate(leaky_rate,nIterations,randomSeedList[j],numOfClasses=8)
#        pdb.set_trace()
        fixedWidthAccuracyMat[j,i,0:nIterations]=torch.FloatTensor(tmp[0])
        fixedWidthleakyRateMat[j,i,0:nIterations]=torch.FloatTensor(tmp[1]).t()
        i=i+1


# Saving the results:
with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNWidthLeakyRateLrnsCOIL100.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([fixedWidthAccuracyMat, fixedWidthleakyRateMat], f)
    # f.close()
# # Getting back the objects:
# with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNWidthLeakyRateLrnsCOIL100.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     fixedWidthAccuracyMat, fixedWidthleakyRateMat = pickle.load(f)














































# import torch
# import pickle
# import numpy as np
# import scipy.stats
# import matplotlib.pyplot as plt
# from loadPretrainedclassifierAndIncorporateitintoESNForRotatingObjLeakyRatechangesCOIL100 import ESNRotatingObjectLrnLeakyRate
# from loadPretrainedclassifierAndIncorporateitintoESNForFixedWidthLeakyRatechangesCOIL100 import ESNFixedWidthLrnLeakyRate
# from tqdm import tqdm
# import pdb
# numberOfRealizations=30
# intialLeakyRateVec=[.25,.5,.75]
# nIterations=18



# rotatObjleakyRateMat=torch.empty(numberOfRealizations,len(intialLeakyRateVec),nIterations)
# rotatObjleakyRateMat[:]=np.nan

# fixedWidthleakyRateMat=torch.empty(numberOfRealizations,len(intialLeakyRateVec),nIterations)
# fixedWidthleakyRateMat[:]=np.nan



# rotatObjAccuracyMat=torch.empty(numberOfRealizations,len(intialLeakyRateVec),nIterations)
# rotatObjAccuracyMat[:]=np.nan

# fixedWidthAccuracyMat=torch.empty(numberOfRealizations,len(intialLeakyRateVec),nIterations)
# fixedWidthAccuracyMat[:]=np.nan



# def genrtIntegerSeeds(howManyNumbs):
#     # generate random integer values
#     from random import seed
#     from random import randint
#     # seed random number generator
#     seed(1)
#     randomSeedList=[]
#     # generate some integers
#     for i in range(howManyNumbs):
#     	randomSeedList.append(randint(10000000, 900000000))
    
#     return randomSeedList

# randomSeedList=genrtIntegerSeeds(numberOfRealizations)

# # for j in tqdm(range(len(randomSeedList))):
# #     i=0
# #     for leaky_rate in intialLeakyRateVec:
# #         tmp=ESNRotatingObjectLrnLeakyRate(leaky_rate,nIterations,randomSeedList[j])
# #         rotatObjAccuracyMat[j,i,0:nIterations]=torch.FloatTensor(tmp[0])
# #         rotatObjleakyRateMat[j,i,0:nIterations]=torch.FloatTensor(tmp[1]).t()
# #         i=i+1






# # # Saving the results:
# # with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ESNObjLeakyRateLrnsCOIL100.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
# #     pickle.dump([rotatObjAccuracyMat, rotatObjleakyRateMat], f)
# #     # f.close()
# # # # Getting back the objects:
# # with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ESNObjLeakyRateLrnsCOIL100.pkl','rb') as f:  # Python 3: open(..., 'rb')
# #     rotatObjAccuracyMat, rotatObjleakyRateMat = pickle.load(f)







# for j in tqdm(range(len(randomSeedList))):
#     i=0
#     for leaky_rate in intialLeakyRateVec:
#         tmp=ESNFixedWidthLrnLeakyRate(leaky_rate,nIterations,randomSeedList[j])
# #        pdb.set_trace()
#         fixedWidthAccuracyMat[j,i,0:nIterations]=torch.FloatTensor(tmp[0])
#         fixedWidthleakyRateMat[j,i,0:nIterations]=torch.FloatTensor(tmp[1]).t()
#         i=i+1


# # Saving the results:
# with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNWidthLeakyRateLrnsCOIL100.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([fixedWidthAccuracyMat, fixedWidthleakyRateMat], f)
#     # f.close()
# # # Getting back the objects:
# # with open('/N/u/aalipour/Karst/2ndYearProject/2020Wave/code/ReviewersAsked/code/ESNWidthLeakyRateLrnsCOIL100.pkl','rb') as f:  # Python 3: open(..., 'rb')
# #     fixedWidthAccuracyMat, fixedWidthleakyRateMat = pickle.load(f)




# #plt.figure()
# #plt.plot(resultsFixedWidth)
# #plt.title('Network Performance- Width Classification')
# #plt.xlabel('leaky rate')
# #plt.ylabel('Ratio Of correct responses to Number Of Images')
# #plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
# #plt.show()


# #i=0
# #for leaky_rate in leaky_rateVec:
# #    resultsFixedOri[i]=ESNFixedOri(leaky_rate)
# #    i=i+1
# #
# #plt.figure()
# #plt.plot(resultsFixedOri)
# #plt.title('Network Performance- Orientation Classification')
# #plt.xlabel('leaky rate')
# #plt.ylabel('Ratio Of correct responses to Number Of Images')
# #plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
# #plt.show()

# #############################################################################

# # fixedWidthleakyRateMat=fixedWidthleakyRateMat.numpy()
# # fixedWidthAccuracyMat=fixedWidthAccuracyMat.numpy()
# # rotatObjAccuracyMat=rotatObjAccuracyMat.numpy()
# # rotatObjleakyRateMat=rotatObjleakyRateMat.numpy()

        
# # import numpy as np
# # objSEM=np.empty([3,nIterations])
# # obj=rotatObjleakyRateMat
# # nanValuesObj=np.argwhere(np.isnan(obj))
# # if nanValuesObj.size!=0:
# #     obj=np.delete(obj,nanValuesObj[0,0],axis=0)
# # for i in range(obj.shape[1]):
# #     for j in range(obj.shape[2]):
# #         objSEM[i,j]=scipy.stats.sem(obj[:,i,j])

# # objMean=np.empty([3,nIterations])
# # for i in range(obj.shape[1]):
# #     for j in range(obj.shape[2]):
# #         objMean[i,j]=(np.mean(obj[:,i,j]))
     

       
# # import numpy as np
# # objAccSEM=np.empty([3,nIterations])
# # objAcc=rotatObjAccuracyMat
# # for i in range(objAcc.shape[1]):
# #     for j in range(objAcc.shape[2]):
# #         objAccSEM[i,j]=scipy.stats.sem(objAcc[:,i,j])

# # objAccMean=np.empty([3,nIterations])
# # for i in range(objAcc.shape[1]):
# #     for j in range(objAcc.shape[2]):
# #         objAccMean[i,j]=(np.mean(objAcc[:,i,j]))

        
# # import numpy as np
# # widthSEM=np.empty([3,nIterations])
# # widthSEM[:]=np.nan
# # width=fixedWidthleakyRateMat
# # nanValuesWidth=np.argwhere(np.isnan(width))
# # if nanValuesWidth.size!=0:
# #     width=np.delete(obj,nanValuesWidth[0,0],axis=0)
    
# # for i in range(width.shape[1]):
# #     for j in range(width.shape[2]):
# #       widthSEM[i,j]=(scipy.stats.sem(width[:,i,j]))

# # widthMean=np.empty([3,nIterations])
# # widthMean[:]=np.nan
# # for mm in range(width.shape[1]):
# #     for kk in range(width.shape[2]):
# #         widthMean[mm,kk]=(np.mean(width[:,mm,kk]))

       

# # widthAccSEM=np.empty([3,nIterations])
# # widthAccSEM[:]=np.nan
# # widthAcc=fixedWidthAccuracyMat

    
# # for i in range(widthAcc.shape[1]):
# #     for j in range(widthAcc.shape[2]):
# #       widthAccSEM[i,j]=(scipy.stats.sem(widthAcc[:,i,j]))

# # widthAccMean=np.empty([3,nIterations])
# # widthAccMean[:]=np.nan
# # for mm in range(widthAcc.shape[1]):
# #     for kk in range(widthAcc.shape[2]):
# #         widthAccMean[mm,kk]=(np.mean(widthAcc[:,mm,kk]))   
     

# # # scatterplot
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.cm as cm

# # # Create data
# # g25obj = (objAccMean[0,:],objMean[0,:])
# # g50obj = (objAccMean[1,:],objMean[1,:])
# # g75obj = (objAccMean[2,:],objMean[2,:])
# # g25width = (widthAccMean[0,:],widthMean[0,:])
# # g50width = (widthAccMean[1,:],widthMean[1,:])
# # g75width = (widthAccMean[2,:],widthMean[2,:])
# # data = (g25obj, g50obj, g75obj,g25width,g50width,g75width)
# # colorsObj = ("cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","cornflowerblue","royalblue","royalblue","royalblue","royalblue","b","b","b","mediumblue","mediumblue","darkblue","k")
# # colorsWidth=("lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","lightcoral","indianred","indianred","indianred","indianred","brown","brown","brown","firebrick","firebrick","maroon","k")

# # # Create plot
# # fig = plt.figure()
# # #plt.title('Leaky Rate VS. Classification Accuracy')

# # ax = fig.add_subplot(1, 3, 1)
# # ax.tick_params(labelsize=16) 
# # ax.scatter(g25obj[0], g25obj[1], alpha=0.8, c=colorsObj, edgecolors='none', s=30,label='Object classification Forget Rate')
# # ax.scatter(g25width[0], g25width[1], alpha=0.8, c=colorsWidth, edgecolors='none', s=30,label='Width classification Forget Rate')
# # xy_line = (0.25, 0.25)
# # ax.plot(xy_line, 'k--', label='Initial Forget Rate')
# # ax.set_xlabel('Network Accuracy',size=24)
# # ax.set_ylabel('Forget Rate',size=24)
# # ax.legend(fontsize=12)
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0.0, 0.8)

# # #for 0.5
# # ax = fig.add_subplot(1, 3, 2)
# # ax.tick_params(labelsize=16) 
# # ax.scatter(g50obj[0], g50obj[1], alpha=0.8, c=colorsObj, edgecolors='none', s=30,label='Object classification Forget Rate')
# # ax.scatter(g50width[0], g50width[1], alpha=0.8, c=colorsWidth, edgecolors='none', s=30,label='Width classification Forget Rate')
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0.0, 0.8)
# # xy_line = (0.5, 0.5)
# # ax.plot(xy_line, 'k--', label='Initial Forget Rate')
# # ax.set_xlabel('Network Accuracy',size=24)
# # #ax.legend(fontsize=14)
# # # for 0.75
# # ax = fig.add_subplot(1, 3, 3)
# # ax.tick_params(labelsize=16) 
# # ax.scatter(g75obj[0], g75obj[1], alpha=0.8, c=colorsObj, edgecolors='none', s=30,label='Object classification Forget Rate')    
# # ax.scatter(g75width[0], g75width[1], alpha=0.8, c=colorsWidth, edgecolors='none', s=30,label='Width classification Forget Rate')
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0.0, 0.8)
# # xy_line = (0.75, 0.75)
# # ax.plot(xy_line, 'k--', label='Initial Forget Rate')
# # ax.set_xlabel('Network Accuracy',size=24)
# # #ax.legend(fontsize=14,loc=7)


# # plt.show()


# ###############################


# # # Create plot for final points
# # fig = plt.figure()
# # # plt.title('Forget Rate VS. Classification Accuracy after 18 Epochs')
# # capSizeVal=2
# # ax = fig.add_subplot(1, 3, 1)
# # ax.tick_params(labelsize=16) 
# # ax.errorbar(g25obj[0][-1], g25obj[1][-1], yerr=objSEM[0,-1],xerr=objAccSEM[0,-1],capsize=capSizeVal,label='Object classification Forget Rate')
# # ax.errorbar(g25width[0][-1], g25width[1][-1], yerr=widthSEM[0,-1],xerr=widthAccSEM[0,-1],capsize=capSizeVal,label='Width classification Forget Rate')
# # xy_line = (0.25, 0.25)
# # ax.plot(xy_line, 'k--', label='Initial Forget Rate')
# # ax.set_xlabel('Network Accuracy',size=24)
# # ax.set_ylabel('Forget Rate',size=24)
# # ax.legend(fontsize=12)
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0.1, 0.4)

# # #for 0.5
# # ax = fig.add_subplot(1, 3, 2)
# # ax.tick_params(labelsize=16) 
# # ax.errorbar(g50obj[0][-1], g50obj[1][-1], yerr=objAccSEM[1,-1],xerr=objAccSEM[1,-1],capsize=capSizeVal,label='Object classification Forget Rate')
# # ax.errorbar(g50width[0][-1], g50width[1][-1], yerr=widthSEM[1,-1],xerr=widthAccSEM[1,-1],capsize=capSizeVal,label='Width classification Forget Rate')
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0.1, 0.8)
# # xy_line = (0.5, 0.5)
# # ax.plot(xy_line, 'k--', label='Initial Forget Rate')
# # ax.set_xlabel('Network Accuracy',size=24)
# # #ax.legend(fontsize=14)
# # # for 0.75
# # ax = fig.add_subplot(1, 3, 3)
# # ax.tick_params(labelsize=16) 
# # ax.errorbar(g75obj[0][-1], g75obj[1][-1],yerr=objAccSEM[2,-1],xerr=objAccSEM[2,-1],capsize=capSizeVal,label='Object classification Forget Rate')    
# # ax.errorbar(g75width[0][-1], g75width[1][-1], yerr=widthSEM[2,-1],xerr=widthAccSEM[2,-1],capsize=capSizeVal,label='Width classification Forget Rate')
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0.1, 0.8)
# # xy_line = (0.75, 0.75)
# # ax.plot(xy_line, 'k--', label='Initial Forget Rate')
# # ax.set_xlabel('Network Accuracy',size=24)
# # #ax.legend(fontsize=14,loc=7)


# # plt.show()



# ################################################################################


# # # create some data
# # xy = np.random.rand(4, 2)
# # xy_line = (0, 1)

# # # set up figure and ax
# # fig, ax = plt.subplots(figsize=(8,8))

# # # create the scatter plots
# # ax.scatter(xy[:, 0], xy[:, 1], c='blue')
# # for point, name in zip(xy, 'ABCD'):
# #     ax.annotate(name, xy=point, xytext=(0, -10), textcoords='offset points',
# #                 color='blue', ha='center', va='center')
# # ax.scatter([0], [1], c='black', s=60)
# # ax.annotate('Perfect Classification', xy=(0, 1), xytext=(0.1, 0.9),
# #             arrowprops=dict(arrowstyle='->'))

# # # create the line
# # ax.plot(xy_line, 'r--', label='Random guess')
# # ax.annotate('Better', xy=(0.3, 0.3), xytext=(0.2, 0.4),
# #             arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
# # ax.annotate('Worse', xy=(0.3, 0.3), xytext=(0.4, 0.2),
# #             arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
# # # add labels, legend and make it nicer
# # ax.set_xlabel('FPR or (1 - specificity)')
# # ax.set_ylabel('TPR or sensitivity')
# # ax.set_title('ROC Space')
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0, 1)
# # ax.legend()
# # plt.tight_layout()
# # plt.savefig('scatter_line.png', dpi=80)


    
# # #import matplotlib.pyplot as plt
# # #
# # #plt.figure()
# # #plt.errorbar(range(50),objMean,yerr=objSTD)
# # #plt.title('Network Performance- Object Classification (60 realizations)')
# # #plt.xlabel('leaky rate')
# # #plt.ylabel('Ratio Of correct responses to Number Of Images')
# # #plt.xticks(ticks=[10, 20, 30, 40, 50 ],labels=[0.20, 0.40, 0.60, 0.80, 1.0 ])
# # #plt.show()

