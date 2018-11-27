from AdjustRotation import *
from segmentation import *
from AnglesHistogram import *
from BlobsDetection import *
from ConnectedComponents import *
from UpperContour import *
from DiskFractal import *
import glob
from sklearn import neighbors
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

all_features = []
labels = []


def training(image, class_num, testing):
    # imageRot = adjust_rotation(image=image)
    # imageCropped = crop_paper(imageRot=imageRot)
    writerLines = segment(image.copy())

    for line in writerLines:
        feature = []

        # feature 1, Angles Histogram
        #feature.extend(AnglesHistogram(line))

        # Calculate Contours
        line = line.astype('uint8')
        _, contours, hierarchy = cv2.findContours(line.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        contours = np.asarray(contours)

        # feature 2, Blobs Detection
        #feature.extend(blobs_features(contours, hierarchy))

        # feature 3, Connected Components
        feature.extend(ConnectedComponents(contours, hierarchy, line.copy()))

        # # feature 4, Lower Contour
        #feature.extend(LowerContourFeatures(line.copy()))
        #
        # # feature 5, Upper Contour
        # feature.extend(UpperContourFeatures(line.copy()))

        # feature 6, Disk Fractal
        #feature.extend(DiskFractal(line.copy()))

        # feature 7, Ellipse Fractal 45
        #feature.extend(EllipseFractal(line.copy(),45))

        # feature 8, Ellipse Fractal 90
        #feature.extend(EllipseFractal(line.copy(),90))

        # feature 9, Ellipse Fractal 135
        #feature.extend(EllipseFractal(line.copy(),135))

        # feature 10, Ellipse Fractal 0
        #feature.extend(EllipseFractal(line.copy(),0))

        if not testing:
            all_features.append(feature)
            labels.append(class_num)
        else:
            all_features_test.append(feature)

        # print(feature)


for filename in glob.glob('iAmDatabase/class1/*.png'):
    image = cv2.imread(filename)
    training(image, 1, False)

for filename in glob.glob('iAmDatabase/class2/*.png'):
    image = cv2.imread(filename)
    training(image, 2, False)

for filename in glob.glob('iAmDatabase/class3/*.png'):
    image = cv2.imread(filename)
    training(image, 3, False)

for filename in glob.glob('iAmDatabase/class4/*.png'):
    image = cv2.imread(filename)
    training(image, 4, False)

for filename in glob.glob('iAmDatabase/class5/*.png'):
    image = cv2.imread(filename)
    training(image, 5, False)

for filename in glob.glob('iAmDatabase/class6/*.png'):
    image = cv2.imread(filename)
    training(image, 6, False)

classifier = neighbors.KNeighborsClassifier(n_neighbors=5)

classifier.fit(all_features, labels)
for filename in glob.glob('iAmDatabase/test/*.png'):
    print(filename)
    image = cv2.imread(filename)
    all_features_test = []
    training(image, -1, True)
    testCase = np.average(all_features_test, axis=0)

    for item in all_features_test:
        print(classifier.predict(np.asarray(item).reshape(1, -1)))
    print(classifier.predict(np.asarray(testCase).reshape(1, -1)))
