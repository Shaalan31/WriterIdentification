import cv2
import numpy as np
import math


# @returns word_dist within_word_dist total_transitions sdW MedianW AverageW threshold
def ConnectedComponents(contours, hierarchy, img, image_shape):
    mask = (hierarchy[:, 3] == 0).astype('int')
    contours = contours[np.where(mask)]
    bounding_rect = np.zeros((len(contours), 6))
    # average h/w
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        bounding_rect[i] = (int(x), int(y), int(w), int(h), int(w * h), int(h / w))
    # h_to_w_ratio
    h_to_w_ratio = np.average(bounding_rect[:, 5], axis=0)

    bounding_rect_sorted = bounding_rect[bounding_rect[:, 0].argsort()]
    iAmDbImageSize = 375 / 8780618
    mask = (bounding_rect_sorted[:, 4] > (iAmDbImageSize * (image_shape[0] * image_shape[1]))).astype('int')
    bounding_rect_sorted = bounding_rect_sorted[np.where(mask)]

    diff_dist_word = np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]
    threshold = np.average(np.abs(np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]))
    word_dist = np.average(diff_dist_word[np.where(diff_dist_word > threshold)])
    # if line consists of only one word
    if math.isnan(word_dist):
        word_dist = 0
    within_word_dist = np.average(np.abs(diff_dist_word[np.where(diff_dist_word < threshold)]))
    if math.isnan(within_word_dist):
        within_word_dist = 0
    total_transitions = 0
    img = img / 255
    for x, y, w, h, a, r in bounding_rect_sorted:
        total_transitions += np.sum(np.abs(np.diff(img[int(y):int(y + h), int(x):int(x + w)])))
    total_transitions /= (2 * bounding_rect_sorted.shape[0])
    sdW = np.sqrt(np.var(bounding_rect_sorted[:, 2]))
    MedianW = np.median(bounding_rect_sorted[:, 2])
    AverageW = np.average(bounding_rect_sorted[:, 2])

    return np.asarray([word_dist, within_word_dist, total_transitions, sdW, MedianW, AverageW, h_to_w_ratio])
