import numpy as np
import cv2
import math


def blobs_features(contours, hierarchy):
    mask = (hierarchy[:, 3] > 0).astype('int')
    contours = contours[np.where(mask)]

    areas = []
    lengths = []
    roundness = []
    form_factors = []

    if len(contours) == 0:
        return [0, 0, 0]
    for contour in contours:
        current_area = cv2.contourArea(contour)
        if current_area == 0:
            continue
        current_length = cv2.arcLength(contour, True)
        current_length_sq = math.pow(current_length, 2)
        areas.append(current_area)
        lengths.append(current_length)
        form_factors.append(4 * current_area * math.pi / current_length_sq)
        roundness.append(current_length_sq / current_area)

    feature_vector = [np.average(areas), np.average(roundness), np.average(form_factors)]
    return feature_vector
