from commonfunctions import *
import scipy.misc as misc
from skimage.filters import gaussian
from scipy import signal


def segment(image):
    image = remove_shadow(image)

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # add a white padding to the image
    # imageGray = cv2.copyMakeBorder(imageGray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Noise removal with gaussian
    imageGray = gaussian(imageGray, 1)

    imageGray[(imageGray * 255 > 200)] = 255
    imageGray[(imageGray * 255 <= 200)] = 0

    # show_images([imageGray])
    # get count of black pixels for each row
    black_count = np.subtract(imageGray.shape[1], np.sum(imageGray * (1/255), axis=1))

    # crop_image
    maxRow = 0
    for i in range(len(black_count) - 1, -1, -1):
        if black_count[i] > 0:
            maxRow = i
            imageGray = imageGray[0:maxRow][:]
            break

    # We're extracting peaks here, assuming minimum peak distance is 150
    peak_indices = signal.find_peaks_cwt(black_count, np.arange(1, 200))

    count = 0
    writer_lines = []
    for i in range(0,len(peak_indices)):
        if i != (len(peak_indices) -1):
            distance = int((peak_indices[i+1] - peak_indices[i]) / 2)
            x = imageGray[peak_indices[i] - distance:peak_indices[i] + distance][0:imageGray.shape[1]]
        else:
            x = imageGray[peak_indices[i] - 75:][0:imageGray.shape[1]]
        if x.shape[0] != 0 and x.shape[1] != 0:
            x = cv2.copyMakeBorder(x, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            writer_lines.append(x)
            misc.imsave('output/image' + str(count) + '.png', x)
            count += 1
    return writer_lines
