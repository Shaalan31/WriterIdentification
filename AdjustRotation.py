import numpy as np
import cv2
import scipy.misc as misc
from skimage.filters import gaussian
from commonfunctions import *
from skimage.filters import threshold_otsu


# Function to detect the edges of the paper to remove the effect of any light
def paper_edge_detection(image):
    imageCopy = image.copy()
    imageResized = cv2.resize(src=imageCopy, dsize=(round((700 / imageCopy.shape[0]) * imageCopy.shape[1]), 700))

    imageGray = cv2.cvtColor(imageResized, cv2.COLOR_RGB2GRAY)

    # bilateral filter for noise removal while keeping edges sharp
    imageBlur = cv2.bilateralFilter(imageGray, 9, 75, 75)
    # show_images([imageBlur])

    # imagesThresh = cv2.adaptiveThreshold(imageBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 59, 4)
    imageBlur *= 255
    threshold = threshold_otsu(imageBlur)
    imageBlur[(imageBlur > threshold)] = 255
    imageBlur[(imageBlur <= threshold)] = 0
    imagesThresh = imageBlur.copy()

    # show_images([imagesThresh])
    imagesThresh = cv2.medianBlur(imagesThresh, 11)
    # show_images([imagesThresh])

    imagesEdDet = cv2.Canny(imagesThresh, 200, 250)
    # show_images([imagesEdDet])

    # closing --> dilation then erosion
    imagesEdDet = cv2.morphologyEx(imagesEdDet, cv2.MORPH_CLOSE, np.ones((3, 3)))  # np.ones((5, 11)

    return imagesEdDet, imageResized


# Function to find the bounding box around a rectangle (paper)
def find_bounding_box(image):
    # Thresholding the image
    imageThresh = np.copy(image)

    # Get image with edge detection
    imageEdgeDet, imageResized = paper_edge_detection(imageThresh)
    # io.imsave('img_edges.png', imageEdgeDet)

    # Get all the countours
    # cv2.RETR_TREE --> (retrieval mode) retrieves all of the contours
    # and reconstructs a full hierarchy of nested contours.
    # cv2.CHAIN_APPROX_SIMPLE -->  (contour approximation mode) compresses horizontal, vertical, and diagonal segments
    # and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
    im, contours, hierarchy = cv2.findContours(np.copy(imageEdgeDet), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour which will be the paper
    maxArea = -1
    index = 0
    rectangle = contours[0]
    for x in range(0, len(contours)):
        contour = contours[x]
        area = cv2.contourArea(contour)
        epsilon = 0.1 * cv2.arcLength(contour, True)
        rect = cv2.approxPolyDP(contour, epsilon, True)
        if (len(rect) == 4):
            if (area > maxArea):
                rectangle = rect
                maxArea = area
                index = x
    contour = contours[index]

    # Draw contour and minAreaRect on the original edge image
    cv2.drawContours(imageResized, [rectangle], -1, (0, 255, 0), 3)
    cv2.drawContours(imageResized, [contour], -1, (255, 0, 0), 3)

    # Get the points of the bounding rect
    height = image.shape[0]
    width = image.shape[1]
    points = rectangle[:, 0]
    x = points[:, 0]
    y = points[:, 1]
    points[:, 0] = x * (height / imageResized.shape[0])
    points[:, 1] = y * (width / imageResized.shape[1])

    # Get the right order of the rectangle points
    rect = np.zeros((4, 2))
    sum = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    # the top-left point will have the smallest sum
    rect[0] = points[np.argmin(sum)]
    # the bottom-right point will have the largest sum
    rect[2] = points[np.argmax(sum)]

    # top-right point will have the smallest difference
    rect[1] = points[np.argmin(diff)]
    # the bottom-left will have the largest difference
    rect[3] = points[np.argmax(diff)]

    # imageGray = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
    # imageGray = np.float32(imageGray)
    # dest = cv2.cornerHarris(imageGray,100,3,0.04)
    # dest = cv2.dilate(dest,None)
    # imageGray[dest>0.01*dest.max()] = 255
    # imageGray[dest<=0.01*dest.max()] = 0
    # show_images([imageGray])
    #
    # imageGray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    # corners = cv2.goodFeaturesToTrack(imageGray, 4, 0.005, 100)
    # corners = np.int0(corners)
    #
    # for corner in corners:
    #     x, y = corner.ravel()
    #     cv2.circle(imageGray, (x, y), 10, 0, -1)
    # show_images([imageGray],['goodFeatures'])
    return rect, imageResized, maxArea


# Function to adjust the rotation of the image
def adjust_rotation(image):
    # get the rotation angle and the bounding box of the paper
    rect, imageBounded, maxArea = find_bounding_box(image=np.copy(image))
    # io.imsave('boundedRot.png', imageBounded)

    if maxArea < 2000:
        return image

    # Get the perspective transformation of the image
    rect = np.float32(rect)

    # Calculate the dimensions of the new Image
    width1 = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    width2 = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))

    height1 = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    height2 = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))

    width = max(int(width1), int(width2))
    height = max(int(height1), int(height2))

    # imageCorners = np.float32([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])
    imageCorners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    perspectiveTransf = cv2.getPerspectiveTransform(rect, imageCorners)
    newImage = cv2.warpPerspective(image, perspectiveTransf, (width, height))
    # io.imsave('final.png', newImage)

    # Crop the margin
    cropped = newImage[170: newImage.shape[0] - 170, 70: newImage.shape[1] - 70]

    return cropped
