from commonfunctions import *
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from itertools import groupby
from operator import itemgetter


def segment(image):
    image = remove_shadow(image)

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal with gaussian
    imageGray = gaussian(imageGray, 1)

    # Thresholding
    imageGray *= 255
    threshold = threshold_otsu(imageGray)
    imageGray[(imageGray > threshold)] = 255
    imageGray[(imageGray <= threshold)] = 0

    imageGray = cv2.erode(imageGray.copy(), np.ones((3, 3)), iterations=1)

    # get count of black pixels for each row
    black_count = np.subtract(imageGray.shape[1], np.sum(imageGray * (1 / 255), axis=1))
    # show_images([imageGray])

    # imageGray = cropPrinted(imageGray.copy(), black_count)
    # show_images([imageGray])

    # crop_image
    maxRow = 0
    for i in range(len(black_count) - 1, -1, -1):
        if (black_count[i] / imageGray.shape[1]) * 100 > 1:
            constant = 0
            if i + 10 <= imageGray.shape[0]:
                constant = 10

            maxRow = i + constant
            imageGray = imageGray[0:maxRow][:]
            break

    writer_lines = []
    line_start = 0
    foundALine = False
    imgName = 0
    for line_index in range(imageGray.shape[0]):
        values, count = np.unique(imageGray[line_index, :], return_counts=True)
        if len(values) == 1:
            foundALine = False
            continue
        countBlack = count[0]
        countWhite = count[1]
        total = countWhite + countBlack
        percentageBlack = (countBlack / total) * 100
        if percentageBlack > 1 and not foundALine:
            foundALine = True
            line_start = line_index
        else:
            if foundALine and percentageBlack < 1:
                if line_index - line_start > 50:
                    line = imageGray[line_start:line_index, :].astype('uint8')
                    line = cv2.copyMakeBorder(line, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    # io.imsave('output/image' + str(imgName) + '.png', line)
                    imgName += 1
                    writer_lines.append(line)
                foundALine = False
    return writer_lines


def cropPrinted(imageGray, blackCount):
    maskIndexed = np.where((blackCount / imageGray.shape[1]) > 0.2)[0]
    tempImage = imageGray[maskIndexed]
    indices = []
    for i in range(0, len(tempImage)):
        array = [x[0] for x in groupby(tempImage[i, :])]
        if (len(array) == 3 or len(array) == 5) and array[0] == 255 and array[len(array) - 1] == 255:
            indices.append(maskIndexed[i])

    indicesNew = []
    for i in range(len(indices) - 2, -1, -1):
        if np.abs(indices[i] - indices[i + 1]) < 10:
            continue
        else:
            indicesNew.append(indices[i] + 1)
            indicesNew.append(indices[i + 1] - 1)
            break
    if len(indicesNew) == 0:
        return imageGray
    return imageGray[indicesNew[0]:indicesNew[1], :]
