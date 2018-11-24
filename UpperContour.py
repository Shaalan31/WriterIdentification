import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
img = cv2.imread('iAmDatabase/line3.png')


img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img[img<180] = 0
img[img>=180] = 1

tester = np.asarray([[0,1,0,1],
                      [0,1,0,1],
                      [0,1,1,0]])

#print(tester.shape[0] - 1 - np.argmin(tester[::-1],axis=0))
#print(tester.shape[0] - 1 - np.argmin(tester,axis=0))

#lower_contour = tester.shape[0] - 1 - np.argmin(tester[::-1],axis=0)
#mask = tester[lower_contour,np.indices(lower_contour.shape)]
#mask = 1 - mask
#print(mask)
#lower_contour = lower_contour[mask[0]==1]
#print(lower_contour)
def LowerContourFeatures(img):
    #lower_contour = img.shape[0] - 1 - np.argmin(img[::-1], axis=0)
    lower_contour = img.shape[0] - 1 - np.argmin(img, axis=0)

    mask = img[lower_contour, np.indices(lower_contour.shape)]
    mask = 1 - mask
    lower_contour = lower_contour[mask[0] == 1]
    diff = np.append(np.asarray([0]),np.diff(lower_contour))
    diff[np.abs(diff)<4]=0
    change = np.cumsum(diff)
    lower_contour = lower_contour - change
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.linspace(0,lower_contour.shape[0],lower_contour.shape[0]),y=lower_contour)
    print(slope,std_err)
    local_max = signal.argrelextrema(lower_contour,np.greater)[0]
    local_min = signal.argrelextrema(lower_contour,np.less)[0]
    max_slope = 0
    min_slope = 0
    minIndex=0
    for i in local_max:
        if(i<5):
            minIndex = 0
        else:
            minIndex = i-5
        slopeMax,_,_,_,_ =  stats.linregress(x=np.linspace(0,i-minIndex+1,i-minIndex+1),y=lower_contour[minIndex:i+1])
        max_slope+=slopeMax
    average_slope_max=slopeMax/len(local_max)
    for i in local_min:
        if(i>len(lower_contour)-5):
            maxIndex = len(lower_contour)
        else:
            maxIndex = i+5
        slopeMin,_,_,_,_ =  stats.linregress(x=np.linspace(0,-i+maxIndex,-i+maxIndex),y=lower_contour[i:maxIndex])
        min_slope+=slopeMin

    average_slope_min=slopeMin/len(local_min)
    max_ratio = len(local_max)/lower_contour.shape[0]
    min_ratio = len(local_min)/lower_contour.shape[0]

    return [slope,std_err,len(local_max),len(local_min),average_slope_max,average_slope_min,max_ratio,min_ratio]
    #fig = plt.figure()
    #ax = fig.add_subplot('111')
    #plt.ylim(0, 100)
    #plt.xlim(0,200)
    #ax.scatter(np.linspace(0,lower_contour.shape[0],lower_contour.shape[0]),lower_contour,s=1)
    #plt.show()

def UpperContourFeatures(img):
    upper_contour = img.shape[0] - 1 - np.argmin(img[::-1], axis=0)
    #lower_contour = img.shape[0] - 1 - np.argmin(img, axis=0)

    mask = img[upper_contour, np.indices(upper_contour.shape)]
    mask = 1 - mask
    upper_contour = upper_contour[mask[0] == 1]
    diff = np.append(np.asarray([0]),np.diff(upper_contour))
    diff[np.abs(diff)<4]=0
    change = np.cumsum(diff)
    upper_contour = upper_contour - change
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=np.linspace(0,upper_contour.shape[0],upper_contour.shape[0]),y=upper_contour)
    print(slope,std_err)
    local_max = signal.argrelextrema(upper_contour,np.greater)[0]
    local_min = signal.argrelextrema(upper_contour,np.less)[0]
    max_slope = 0
    min_slope = 0
    minIndex=0
    for i in local_max:
        if(i<5):
            minIndex = 0
        else:
            minIndex = i-5
        slopeMax,_,_,_,_ =  stats.linregress(x=np.linspace(0,i-minIndex+1,i-minIndex+1),y=upper_contour[minIndex:i+1])
        max_slope+=slopeMax
    average_slope_max=max_slope/len(local_max)
    for i in local_min:
        if(i>len(upper_contour)-5):
            maxIndex = len(upper_contour)
        else:
            maxIndex = i+5
        slopeMin,_,_,_,_ =  stats.linregress(x=np.linspace(0,-i+maxIndex,-i+maxIndex),y=upper_contour[i:maxIndex])
        min_slope+=slopeMin

    average_slope_min=min_slope/len(local_min)
    max_ratio = len(local_max)/upper_contour.shape[0]
    min_ratio = len(local_min)/upper_contour.shape[0]

    return [slope,std_err,len(local_max),len(local_min),average_slope_max,average_slope_min,max_ratio,min_ratio]
    #fig = plt.figure()
    #ax = fig.add_subplot('111')
    #plt.ylim(0, 100)
    #plt.xlim(0,200)
    #ax.scatter(np.linspace(0,lower_contour.shape[0],lower_contour.shape[0]),lower_contour,s=1)
    #plt.show()

