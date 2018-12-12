from segmentation import *
from AnglesHistogram import *
from BlobsDetection import *
from ConnectedComponents import *
from DiskFractal import *
from AdjustRotation import *
import glob
from sklearn import neighbors
import warnings
from itertools import combinations
import time
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global Variables
all_features = np.asarray([])
all_features_class = np.asarray([])
labels = []
temp = []
blob_starting_index = 5
num_training_examples = 0
num_features = 18
num_lines_per_class = 0
num_classes = 159

total_test_cases = 100


def feature_extraction(example):
    example = example.astype('uint8')
    example_copy = example.copy()

    feature = []

    # feature 1, Angles Histogram
    feature.extend(AnglesHistogram(example))

    # Calculate Contours
    _, contours, hierarchy = cv2.findContours(example_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    contours = np.asarray(contours)

    # feature 2, Blobs Detection
    feature.extend(blob_threaded(contours, hierarchy))

    # feature 3, Connected Components
    feature.extend(ConnectedComponents(contours, hierarchy, example_copy))

    # feature 4, Disk Fractal
    feature.extend(DiskFractal(example_copy))

    return np.asarray(feature)


def test(image, clf, mu, sigma):
    all_features_test = np.asarray([])

    if image.shape[0] > 3500:
        image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image.shape[0])))

    # image = adjust_rotation(image=image)
    # show_images([image], ["rotation"])
    writerLines = segment(image)

    num_testing_examples = 0
    for line in writerLines:
        example = feature_extraction(line)
        all_features_test = np.append(all_features_test, example)
        num_testing_examples += 1

    all_features_test = (adjustNaNValues(
        np.reshape(all_features_test, (num_testing_examples, num_features))) - mu) / sigma

    # Predict on each line
    # predictions = []
    # for example in all_features_test:
    #     predictions.append(clf.predict(np.asarray(example).reshape(1, -1)))
    # values, counts = np.unique(np.asarray(predictions), return_counts=True)
    # return values[np.argmax(counts)]
    return clf.predict(np.average(all_features_test, axis=0).reshape(1, -1))


def training(image, class_num):
    global all_features
    global all_features_class
    global labels
    global num_lines_per_class
    global num_training_examples

    image_height = image.shape[0]
    if image_height > 3500:
        image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image_height)))

    # image = adjust_rotation(image=image)
    # show_images([image], ["rotation"])
    writerLines = segment(image)

    num_lines_per_class += len(writerLines)
    for line in writerLines:
        all_features_class = np.append(all_features_class, feature_extraction(line))
        labels.append(class_num)
        num_training_examples += 1

    return np.reshape(all_features_class, (num_lines_per_class, num_features))


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


def featureNormalize(X):
    mean = np.mean(X, axis=0)
    normalized_X = X - mean
    deviation = np.sqrt(np.var(normalized_X, axis=0))
    normalized_X = np.divide(normalized_X, deviation)
    return normalized_X, mean, deviation


def reading_test_cases():
    global all_features
    global temp
    global num_training_examples
    global labels
    global num_lines_per_class
    global all_features_class
    global totalAnswers
    global correctAnswers
    global avgTime
    indices_array = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    for i in range(10, 101):
        indices_array.append(str(i))

    for index in indices_array:
        millis = int(round(time.time() * 1000))
        test_combination = (1, 2, 3)
        for class_number in test_combination:
            num_lines_per_class = 0
            all_features_class = np.asarray([])
            for filename in glob.glob('data/' + index + '/' + str(class_number) + '/*.PNG'):
                print(filename)
                temp = training(cv2.imread(filename), class_number)
            all_features = np.append(all_features,
                                     np.reshape(adjustNaNValues(temp), (1, num_lines_per_class * num_features)))

        # Normalization of features
        all_features, mu, sigma = featureNormalize(np.reshape(all_features, (num_training_examples, num_features)))

        classifier.fit(all_features, labels)
        labels = []
        all_features = []
        num_training_examples = 0
        for filename in glob.glob('data/' + index + '/test.PNG'):
            print(filename)
            # label = int(filename[len(filename) - 5])
            # print(label)
            prediction = test(cv2.imread(filename), classifier, mu, sigma)
            print(prediction)

            # if prediction == label:
            #     correctAnswers += 1
            # else:
            #     file = open("wrngClassified.txt", "a")
            #     file.write(str(test_combination))
            #     file.write('\n')
            #     file.close()
            totalAnswers += 1
            # accuracy = (correctAnswers / totalAnswers) * 100
            # print("Accuracy = ", accuracy, "%")
        avgTime = (int(round(time.time() * 1000)) - millis)
        print("-----------------------------------------------------------------")

        print("Time:")
        print(avgTime/1000 )
        print("-----------------------------------------------------------------")


correctAnswers = 0
totalAnswers = 0
# class_labels = list(range(1, num_classes + 1))
# classCombinations = list(combinations(class_labels, r=3))
avgTime = 0
# classifier = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
classifier = MLPClassifier(solver='lbfgs', max_iter=20000, alpha=1e-16, hidden_layer_sizes=(22,), random_state=1)

for z in range(1, 2):
    avgTime = 0
    correctAnswers = 0
    totalAnswers = 0
    reading_test_cases()

# for test_combination in classCombinations:
#     print(test_combination)
#     millis = int(round(time.time() * 1000))
#     for class_number in test_combination:
#         num_lines_per_class = 0
#         all_features_class = np.asarray([])
#         for filename in glob.glob('Samples/Class' + str(class_number) + '/*.png'):
#             temp = training(cv2.imread(filename), class_number)
#         all_features = np.append(all_features,
#                                  np.reshape(adjustNaNValues(temp), (1, num_lines_per_class * num_features)))
#
#     # Normalization of features
#     all_features, mu, sigma = featureNormalize(np.reshape(all_features, (num_training_examples, num_features)))
#
#     # pca = PCA(n_components=0.99, svd_solver='full', whiten=True).fit(all_features)
#     # all_features_training_trans = pca.transform(all_features)
#
#     classifier.fit(all_features, labels)
#     labels = []
#     all_features = []
#     num_training_examples = 0
#     for class_number in test_combination:
#         for filename in glob.glob('TestCases/testing' + str(class_number) + '.png'):
#             print(filename)
#             prediction = test(cv2.imread(filename), classifier, mu, sigma)
#             print(prediction)
#
#             if prediction == class_number:
#                 correctAnswers += 1
#             else:
#                 file = open("wrngClassified.txt", "a")
#                 file.write(str(test_combination))
#                 file.write('\n')
#                 file.close()
#             totalAnswers += 1
#             accuracy = (correctAnswers / totalAnswers) * 100
#             print("Accuracy = ", accuracy, "%")
#     print((int(round(time.time() * 1000)) - millis) / 60000)
#     avgTime += (int(round(time.time() * 1000)) - millis)
#     print("-----------------------------------------------------------------")
# print("Average Time:")
# print(avgTime / (totalAnswers * 60000))
# print("-----------------------------------------------------------------")
#
# accuracy = (correctAnswers / totalAnswers) * 100
# print(accuracy)
