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
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global Variables
all_features = []
all_features = np.asarray(all_features)
all_features_class = []
all_features_class = np.asarray(all_features_class)
all_features_test = []
all_features_test = np.asarray(all_features_test)
labels = []
temp = []
blob_starting_index = 5
num_training_examples = 0
num_testing_examples = 0
num_features = 18
num_lines_per_class = 0
num_classes = 3


def training(image, class_num, testing):
    global all_features
    global all_features_test
    global all_features_class
    global labels
    global num_lines_per_class
    global num_training_examples
    global num_testing_examples

    # image = adjust_rotation(image=image)

    writerLines = segment(image.copy())

    num_lines_per_class += len(writerLines)
    for line in writerLines:
        feature = []

        # feature 1, Angles Histogram
        feature.extend(AnglesHistogram(line))

        # Calculate Contours
        line = line.astype('uint8')
        _, contours, hierarchy = cv2.findContours(line.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        contours = np.asarray(contours)

        # feature 2, Blobs Detection
        feature.extend(blobs_features(contours, hierarchy,image.shape))

        # feature 3, Connected Components
        feature.extend(ConnectedComponents(contours, hierarchy, line.copy()))

        # # feature 4, Lower Contour
        # feature.extend(LowerContourFeatures(line.copy()))
        #
        # # feature 5, Upper Contour
        # feature.extend(UpperContourFeatures(line.copy()))

        # feature 6, Disk Fractal
        feature.extend(DiskFractal(line.copy()))

        # feature 7, Ellipse Fractal 45
        # feature.extend(EllipseFractal(line.copy(),45))

        # feature 8, Ellipse Fractal 90
        # feature.extend(EllipseFractal(line.copy(),90))

        # feature 9, Ellipse Fractal 135
        # feature.extend(EllipseFractal(line.copy(),135))

        # feature 10, Ellipse Fractal 0
        # feature.extend(EllipseFractal(line.copy(),0))

        feature = np.asarray(feature)

        if not testing:
            all_features_class = np.append(all_features_class, feature)
            labels.append(class_num)
            num_training_examples += 1
        else:
            all_features_test = np.append(all_features_test, feature)
            num_testing_examples += 1

    if not testing:
        return np.reshape(all_features_class, (num_lines_per_class, num_features))
    else:
        all_features_test = np.reshape(all_features_test, (num_testing_examples, num_features))
        return adjustNaNValues(all_features_test)


def adjustNaNValues(writer_features):
    for i in range(num_features):
        feature = writer_features[:, i]
        is_nan_mask = np.isnan(feature)
        non_nan_indices = np.where(np.logical_not(is_nan_mask))[0]
        nan_indices = np.where(is_nan_mask)[0]

        if len(nan_indices) > 0:
            if len(non_nan_indices) == 0:
                feature_mean = 0
            else:
                feature_mean = np.mean(feature[non_nan_indices])
            writer_features[nan_indices, i] = feature_mean
    return writer_features


def performPCA(allFeatures):
    allFeaturesY = allFeatures - np.mean(allFeatures, axis=0)
    print(allFeaturesY)
    pca = PCA(n_components=18)
    components = pca.fit_transform(allFeaturesY)
    print(components)
    return components


for class_number in range(1, num_classes + 1):
    num_lines_per_class = 0
    all_features_class = []
    all_features_class = np.asarray(all_features_class)
    print(class_number)
    for filename in glob.glob('TestCases/Class' + str(class_number) + '/*.png'):
        image = cv2.imread(filename)
        temp = training(image, class_number, False)
    temp = adjustNaNValues(temp)
    temp = np.reshape(temp, (1, num_lines_per_class * num_features))
    all_features = np.append(all_features, temp)

all_features = np.reshape(all_features, (num_training_examples, num_features))
# performPCA(all_features)
classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
classifier.fit(all_features, labels)

for filename in glob.glob('TestCases/testing/*.png'):
    print(filename)
    image = cv2.imread(filename)
    all_features_test = []
    all_features_test = np.asarray(all_features_test)
    num_testing_examples = 0
    temp = training(image, -1, True)
    temp = adjustNaNValues(temp)
    testCase = np.average(temp, axis=0)
    print(classifier.predict(np.asarray(testCase).reshape(1, -1)))
