import numpy as np
import cv2
import scipy.misc as misc
from skimage.filters import gaussian


def find_bounding_box(image):
    # Convert to grayscale
    imageGray = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2GRAY)
    # imageGray = rgb2gray(image) * 255

    # Thresholding the image
    imageThresh = np.copy(imageGray)
    imageThresh[imageThresh > 125] = 255
    imageThresh[imageThresh <= 125] = 0

    # erosion then dilation to fill the gaps and remove the noise
    imageErode = cv2.erode(imageThresh, np.ones((3, 3)), iterations=5)
    imageErodDil = cv2.dilate(imageErode, np.ones((3, 3)), iterations=5)

    # Get all the countours
    # cv2.RETR_TREE --> (retrieval mode) retrieves all of the contours
    # and reconstructs a full hierarchy of nested contours.
    # cv2.CHAIN_APPROX_SIMPLE -->  (contour approximation mode) compresses horizontal, vertical, and diagonal segments
    # and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
    im, contours, hierarchy = cv2.findContours(imageErodDil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour which will be the paper
    maxArea = -1
    index = 0
    for x in range(0, len(contours)):
        contour = contours[x]
        area = cv2.contourArea(contour)
        if (area > maxArea):
            maxArea = area
            index = x
    contour = contours[index]

    # Get the minimum area rectangle with the biggest contour
    rectangle = cv2.minAreaRect(contour)
    angleRot = rectangle[-1]
    boxRect = cv2.boxPoints(rectangle)
    boxRect = np.int0(boxRect)

    # Adjust the angle returned from minAreaRect
    # if the width is smaller than the heignt
    if rectangle[1][0] < rectangle[1][1]:
        angleRot = rectangle[-1] + 180
    else:
        angleRot = rectangle[-1] + 90

    if angleRot < 90:
        angleRot = 90 + rectangle[-1]
        # if the angle exceeds 90 then substract 180
    else:
        angleRot = angleRot - 180

    # Draw contour and minAreaRect on the original image
    imageCopy = np.copy(image)
    cv2.drawContours(imageCopy, [boxRect], -1, (0, 255, 0), 2)
    cv2.drawContours(imageCopy, [contour], -1, (255, 0, 0), 2)

    return angleRot, boxRect, imageCopy


# Adjust the rotation of the image
def adjust_rotation(image):
    # get the rotation angle and the bounding box of the paper
    angleRot, boxRect, imageBounded = find_bounding_box(image=np.copy(image))
    misc.imsave('boundedRot.png', imageBounded)

    # rotate the image with the angle
    height = image.shape[0]
    width = image.shape[1]
    center = (width / 2, height / 2)
    rotMatrix = cv2.getRotationMatrix2D(center, angleRot, 1.0)
    # imageRot = cv2.warpAffine(image, rotMatrix, (width, height), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
    imageRot = cv2.warpAffine(image, rotMatrix, (width, height))

    return imageRot


def crop_paper(imageRot):
    # get the bounding box of the paper on the rotated image
    angleRot, boxRect, imageBounded = find_bounding_box(image=np.copy(imageRot))
    misc.imsave('boundedCrop.png', imageBounded)

    # Crop the image on the rotated one
    x1 = boxRect[:, 0].min() + 100
    y1 = boxRect[:, 1].min() + 100
    x2 = boxRect[:, 0].max() - 100
    y2 = boxRect[:, 1].max() - 100
    imageCropped = imageRot[y1: y2, x1: x2]

    return imageCropped


