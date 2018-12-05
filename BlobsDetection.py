import numpy as np
import cv2
import math
from skimage.feature import blob_dog, blob_log, blob_doh
import matplotlib.pyplot as plt
from math import sqrt
from skimage.color import rgb2gray
from commonfunctions import *


def blobs_features(contours, hierarchy, paperShape):
    mask = (hierarchy[:, 3] > 0).astype('int')
    contours = contours[np.where(mask)]

    areas = []
    lengths = []
    roundness = []
    form_factors = []

    for contour in contours:
        current_area = cv2.contourArea(contour)  # / (paperShape[0] * paperShape[1])
        if current_area == 0:
            continue
        current_length = cv2.arcLength(contour, True)  # /(2 * (paperShape[0] + paperShape[1]))
        current_length_sq = math.pow(current_length, 2)
        areas.append(current_area)
        lengths.append(current_length)
        form_factors.append(4 * current_area * math.pi / current_length_sq)
        roundness.append(current_length_sq / current_area)

    feature_vector = [np.average(areas), np.average(roundness), np.average(form_factors)]
    return feature_vector


def get_blobs(image_gray, contours, hierarchy):
    mask = (hierarchy[:, 3] > 0).astype('int')
    contours = contours[np.where(mask)]
    contimg = cv2.drawContours(image_gray.copy(), contours, -1, (255, 0, 0), 2)
    show_images([contimg])

    blobs_log = blob_log(image_gray, max_sigma=10, num_sigma=10, threshold=.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=10, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=10, threshold=.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image_gray, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=0.5, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()
