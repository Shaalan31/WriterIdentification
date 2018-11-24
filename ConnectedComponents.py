import cv2
import numpy as np
import math

# @returns word_dist within_word_dist total_transitions sdW MedianW AverageW threshold
def ConnectedComponents(contours, hierarchy, img):
    mask = (hierarchy[:, 3] == 0).astype('int')
    contours = contours[np.where(mask)]
    bounding_rect = np.zeros((len(contours), 5))
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        bounding_rect[i] = (x, y, w, h, w * h)

    bounding_rect_sorted = bounding_rect[bounding_rect[:, 0].argsort()]
    mask = (bounding_rect_sorted[:, 4] > 375).astype('int')
    bounding_rect_sorted = bounding_rect_sorted[np.where(mask)]
    # imgReal = cv2.imread('iAmDatabase/line1.png')

    for x, y, w, h, a in bounding_rect_sorted:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        # imgReal = cv2.rectangle(imgReal,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imwrite('iAmDatabase/boxes_img76.png', imgReal)

    # print(np.diff(bounding_rect_sorted,axis=0))
    diff_dist_word = np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]
    threshold = np.average(np.abs(np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]))
    word_dist = np.average(diff_dist_word[np.where(diff_dist_word > threshold)])
    #if line consists of only one word
    if math.isnan(word_dist):
        word_dist = 0
    within_word_dist = np.average(np.abs(diff_dist_word[np.where(diff_dist_word < threshold)]))

    total_transitions = 0
    img = img / 255
    for x, y, w, h, a in bounding_rect_sorted:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        total_transitions += np.sum(np.abs(np.diff(img[y:y + h, x:x + w])))
    total_transitions /= (2 * bounding_rect_sorted.shape[0])

    sdW = np.sqrt(np.var(bounding_rect_sorted[:, 2]))
    MedianW = np.median(bounding_rect_sorted[:, 2])
    AverageW = np.average(bounding_rect_sorted[:, 2])

    return np.asarray([word_dist, within_word_dist, total_transitions, sdW, MedianW, AverageW, threshold])
