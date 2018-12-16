import numpy as np
import cv2
from scipy import stats
import scipy.ndimage as ndimage


def DiskFractal(img, loops=25):
    arr = np.zeros((loops, 2))
    arr[1] = ([np.log(1), np.log(np.sum(255 - img) / 255) - np.log(1)])
    for x in range(2, loops):
        img_dilate = cv2.erode(img.copy(), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * x - 1, 2 * x - 1)),
                               iterations=1)
        arr[x] = ([np.log(x), np.log(np.sum(255 - img_dilate) / 255) - np.log(x)])

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


def EllipseFractal(img, angle, loops=25):
    arr = np.zeros((loops, 2))
    arr[1] = ([np.log(1), np.log(np.sum(255 - img) / 255) - np.log(1)])

    for x in range(2, loops):
        ellipse_mask = ndimage.rotate(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * x - 1, 2 * (2 * x - 1))),
                                      float(angle))
        img_dilate = cv2.erode(img.copy(), ellipse_mask,
                               iterations=1)
        arr[x] = ([np.log(x), np.log(np.sum(255 - img_dilate) / 255) - np.log(x)])

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
