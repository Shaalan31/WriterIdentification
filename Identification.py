from segmentation import *
from AnglesHistogram import *
from BlobsDetection import *
from ConnectedComponents import *
from DiskFractal import *
from AdjustRotation import *
import glob
from sklearn import neighbors
import warnings
from sklearn.decomposition import PCA
from itertools import combinations
import time
import pandas as pd

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
num_features = 17
num_lines_per_class = 0
num_classes = 19


def training(image, class_num, testing):
    global all_features
    global all_features_test
    global all_features_class
    global labels
    global num_lines_per_class
    global num_training_examples
    global num_testing_examples

    if image.shape[0] > 3500:
        image = cv2.resize(src=image.copy(), dsize=(3500, round((3500 / image.shape[1]) * image.shape[0])))

    # image = adjust_rotation(image=image)
    # show_images([image], ["rotation"])
    writerLines = segment(image.copy())

    num_lines_per_class += len(writerLines)
    for line in writerLines:
        feature = []
        # show_images([line], ["line"])
        # feature 1, Angles Histogram
        feature.extend(AnglesHistogram(line))

        # Calculate Contours
        line = line.astype('uint8')
        _, contours, hierarchy = cv2.findContours(line.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        contours = np.asarray(contours)

        # feature 2, Blobs Detection
        feature.extend(blobs_features(contours, hierarchy, image.shape))

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
        if len(np.where(np.asarray(is_nan_mask))[0]) == 0:
            continue

        non_nan_indices = np.where(np.logical_not(is_nan_mask))[0]
        nan_indices = np.where(is_nan_mask)[0]

        if len(non_nan_indices) == 0:
            feature_mean = 0
        else:
            feature_mean = np.mean(feature[non_nan_indices])
        writer_features[nan_indices, i] = feature_mean

    return writer_features


# def adjustNaNValuesVectorized(writer_features):
#     is_nan_mask = np.isnan(writer_features)
#     non_nan_indices = np.logical_not(is_nan_mask)
#     nan_indices = is_nan_mask
#     if len(nan_indices) > 0:
#         if len(non_nan_indices) == 0:
#             feature_mean = 0
#         else:
#             feature_mean = np.mean(writer_features[non_nan_indices])
#         writer_features[nan_indices] = feature_mean
#     return writer_features


def featureNormalize(X):
    mean = np.mean(X, axis=0)
    normalized_X = X - np.mean(X, axis=0)
    variances = np.var(normalized_X, axis=0)
    deviation = np.sqrt(variances)
    normalized_X = np.divide(normalized_X, deviation)
    return normalized_X, mean, deviation


# for class_number in range(1, num_classes + 1):
#     num_lines_per_class = 0
#     all_features_class = []
#     all_features_class = np.asarray(all_features_class)
#     print(class_number)
#     for filename in glob.glob('TestCases/Class' + str(class_number) + '/*.png'):
#         image = cv2.imread(filename)
#         temp = training(image, class_number, False)
#     temp = adjustNaNValues(temp)
#     temp = np.reshape(temp, (1, num_lines_per_class * num_features))
#     all_features = np.append(all_features, temp)
#
# all_features = np.reshape(all_features, (num_training_examples, num_features))
# # performPCA(all_features)
# classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
# classifier.fit(all_features, labels)
#
# for filename in glob.glob('TestCases/testing/*.png'):
#     print(filename)
#     image = cv2.imread(filename)
#     all_features_test = []
#     all_features_test = np.asarray(all_features_test)
#     num_testing_examples = 0
#     temp = training(image, -1, True)
#     temp = adjustNaNValues(temp)
#     testCase = np.average(temp, axis=0)
#     print(classifier.predict(np.asarray(testCase).reshape(1, -1)))

correctAnswers = 0
totalAnswers = 0
class_labels = [x for x in range(2, num_classes + 1)]
classCombinations = list(combinations(class_labels, r=3))
avgTime = 0
muClasses = []

for test_combination in classCombinations:
    print(test_combination)
    millis = int(round(time.time() * 1000))
    mu = 0
    sigma = 0
    for class_number in test_combination:
        num_lines_per_class = 0
        all_features_class = []
        all_features_class = np.asarray(all_features_class)
        for filename in glob.glob('Samples/Class' + str(class_number) + '/*.png'):
            image = cv2.imread(filename)
            temp = training(image, class_number, False)
        temp = adjustNaNValues(temp)
        temp = np.reshape(temp, (1, num_lines_per_class * num_features))
        all_features = np.append(all_features, temp)

    # Normalization of features
    all_features = np.reshape(all_features, (num_training_examples, num_features))
    all_features, mu, sigma = featureNormalize(all_features)

    # pca = PCA(n_components=0.99, svd_solver='full', whiten=True).fit(all_features)
    # all_features_training_trans = pca.transform(all_features)

    classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
    classifier.fit(all_features, labels)
    labels = []
    all_features = []
    num_training_examples = 0
    localCorrect = 0
    for class_number in test_combination:
        for filename in glob.glob('TestCases/testing' + str(class_number) + '.png'):
            print(filename)
            image = cv2.imread(filename)
            all_features_test = []
            all_features_test = np.asarray(all_features_test)
            num_testing_examples = 0
            temp = training(image, -1, True)
            temp = (temp - mu) / sigma
            # temp_transform = pca.transform(temp)
            testCase = np.average(temp, axis=0)
            prediction = classifier.predict(np.asarray(testCase).reshape(1, -1))

            print(prediction)
            if prediction == class_number:
                localCorrect += 1
                correctAnswers += 1
            else:
                file = open("wrngClassified.txt", "a")
                file.write(str(test_combination))
                file.write('\n')
                file.close()
            totalAnswers += 1
            accuracy = (correctAnswers / totalAnswers) * 100
            print("Accuracy = ", accuracy, "%")
    print((int(round(time.time() * 1000)) - millis) / 60000)
    avgTime += (int(round(time.time() * 1000)) - millis)
    print("-----------------------------------------------------------------")
print("Average Time:")
print(avgTime / (totalAnswers * 60000))
print("-----------------------------------------------------------------")

accuracy = (correctAnswers / totalAnswers) * 100
print(accuracy)
