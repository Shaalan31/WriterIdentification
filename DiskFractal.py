import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal


#
#
# img = cv2.imread('iAmDatabase/second.png')
#
#
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img[img<180] = 0
# img[img>=180] = 255
# img_dilate = cv2.dilate(img.copy(),np.ones((3,3),np.uint8),iterations=1)
#
# cv2.imwrite('iAmDatabase/boxes_img76.png', img_dilate)
#
# arr = np.zeros((25,2))
# arr[1] = ([ np.log(1),np.log(np.sum(255 - img) / 255) - np.log(1)])
#
# for x in range(2,25):
#     img_dilate = cv2.erode(img.copy(),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*x-1,2*x-1)), iterations=1)
#     arr[x]=([np.log(x),np.log(np.sum(255-img_dilate)/255) - np.log(x)])
#     cv2.imwrite('iAmDatabase/boxes_'+str(x)+'.png', img_dilate)
#
# print(arr[:,0])
#
# print(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
#
# fig = plt.figure()
# ax = fig.add_subplot('111')
# plt.ylim(np.min(arr[1:,1])-0.5,np.max(arr[1:,1])+0.5)
# plt.xlim(-0.3,3.5)
# ax.scatter(x=arr[1:,0],y=arr[1:,1],s=2)
# #plt.show()
#
# error=999
# slope=[0,0,0]
# result=[0,0]
# for x in range(2,23):
#     for y in range(x+2,24):
#         print((x,y))
#         first = arr[1:x+1,:]
#         second = arr[x+1:y+1,:]
#         third = arr[y+1:25,:]
#         slope1, _, _, _, std_err1 = stats.linregress(x=first[:,0], y=first[:,1])
#         slope2, _, _, _, std_err2 = stats.linregress(x=second[:, 0], y=second[:, 1])
#         slope3, _, _, _, std_err3 = stats.linregress(x=third[:, 0], y=third[:, 1])
#
#         if error > std_err1+std_err2+std_err3:
#             error = std_err1+std_err2+std_err3
#             slope = [slope1,slope2,slope3]
#             result = [x,y]
#
# print(slope)
# print(result)


def DiskFractal(img, loops=25):
    arr = np.zeros((loops, 2))
    arr[1] = ([np.log(1), np.log(np.sum(255 - img) / 255) - np.log(1)])
    for x in range(2, loops):
        img_dilate = cv2.erode(img.copy(), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * x - 1, 2 * x - 1)),
                               iterations=1)
        arr[x] = ([np.log(x), np.log(np.sum(255 - img_dilate) / 255) - np.log(x)])
        cv2.imwrite('iAmDatabase/boxes_' + str(x) + '.png', img_dilate)
    error = 999
    slope = [0, 0, 0]
    loops = int(loops)
    for x in range(2, loops - 2):
        for y in range(x + 2, loops - 1):
            first = arr[1:x + 1, :]
            second = arr[x + 1:y + 1, :]
            third = arr[y + 1:loops, :]
            slope1, _, _, _, std_err1 = stats.linregress(x=first[:, 0], y=first[:, 1])
            slope2, _, _, _, std_err2 = stats.linregress(x=second[:, 0], y=second[:, 1])
            slope3, _, _, _, std_err3 = stats.linregress(x=third[:, 0], y=third[:, 1])

            if error > std_err1 + std_err2 + std_err3:
                error = std_err1 + std_err2 + std_err3
                slope = [slope1, slope2, slope3]

    return slope

# print(DiskFractal(img,25))
