from commonfunctions import *
import skimage.filters as filters


def AnglesHistogram(image):
    values, count = np.unique(image, return_counts=True)
    countBlack = count[0]

    sob_img_v = np.multiply(filters.sobel_v(image), 255)
    sob_img_h = np.multiply(filters.sobel_h(image), 255)

    # Getting angles in radians
    angles = np.arctan2(sob_img_v, sob_img_h)
    angles = np.multiply(angles, (180 / math.pi))
    angles = np.round(angles)

    anglesHist = []
    angle1 = 10
    angle2 = 40

    while angle2 < 180:
        anglesCopy = angles.copy()
        anglesCopy[np.logical_or(anglesCopy < angle1, anglesCopy > angle2)] = 0
        anglesCopy[np.logical_and(anglesCopy >= angle1, anglesCopy <= angle2)] = 1
        anglesHist.append(np.sum(anglesCopy))
        angle1 += 30
        angle2 += 30

    return np.divide(anglesHist, countBlack)
