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
import random


warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global Variables
num_features = 18
num_lines_per_class = 0
num_classes = 165
training_dict = {}
testing_dict = {}


def process_training_data():
    global num_lines_per_class

    for class_number in range(1, num_classes + 1):
        temp = np.asarray([])
        num_lines_per_class = 0
        for filename in glob.glob('Samples/Class' + str(class_number) + '/*.png'):
            print(filename)
            temp = np.append(temp, training(cv2.imread(filename)))
        training_dict[class_number] = adjustNaNValues(np.reshape(temp, (num_lines_per_class, num_features)))


def process_test_data():
    for class_number in range(1, num_classes + 1):
        temp = np.asarray([])
        for filename in glob.glob('TestCases/testing' + str(class_number) + '.png'):
            temp = test(cv2.imread(filename))
        testing_dict[class_number] = temp


def start():
    correct_answers = 0
    total_answers = 0
    class_labels = list(range(1, num_classes + 1))
    classCombinations = list(combinations(class_labels, r=3))
    classifier = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs=-1)


    total = len(classCombinations)
    avg = np.array([])
    for i in range(0, 1000):
        accuracy=0
        for j in range(1, 101):
            test_combination = classCombinations[random.randint(0, total - 1)]
            all_features = np.asarray([])
            labels = np.asarray([])
            num_training_examples = 0

            #print(test_combination)
            for class_number in test_combination:
                num_current_examples = len(training_dict[class_number])
                labels = np.append(labels, np.full(shape=(1, num_current_examples), fill_value=class_number))
                num_training_examples += num_current_examples
                all_features = np.append(all_features,
                                         np.reshape(training_dict[class_number].copy(),
                                                    (1, num_current_examples * num_features)))

            all_features = np.reshape(all_features, (num_training_examples, num_features))
            all_features, mu, sigma = featureNormalize(all_features)

            classifier.fit(all_features, labels)

            classNum = random.randint(0, 2)
            #print(test_combination[classNum])
            test_vector = (testing_dict[test_combination[classNum]]).copy()
            test_vector = (test_vector - mu) / sigma
            prediction = classifier.predict(test_vector.reshape(1, -1))
            #print(prediction)

            if prediction == test_combination[classNum]:
                correct_answers += 1
            else:
                file = open("wrngClassified.txt", "a")
                file.write(str(test_combination))
                file.write('\n')
                file.close()
            total_answers += 1
            accuracy = (correct_answers / total_answers) * 100
            #print("Accuracy = ", accuracy, "%")
        avg = np.append(avg,accuracy)
        correct_answers=0
        total_answers=0
        print(i)
        np.savetxt("fasttest.csv", avg, delimiter=",")




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


def test(image):
    all_features_test = np.asarray([])

    if image.shape[0] > 3500:
        image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image.shape[0])))

    # image = adjust_rotation(image=image)
    writerLines = segment(image)

    num_testing_examples = len(writerLines)
    for line in writerLines:
        all_features_test = np.append(all_features_test, feature_extraction(line))

    return np.average(adjustNaNValues(np.reshape(all_features_test, (num_testing_examples, num_features))), axis=0)


def training(image):
    global num_lines_per_class

    image_height = image.shape[0]
    if image_height > 3500:
        image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image_height)))

    # image = adjust_rotation(image=image)
    writerLines = segment(image)

    num_lines = len(writerLines)
    num_lines_per_class += num_lines

    all_features_class = np.asarray([])
    for line in writerLines:
        all_features_class = np.append(all_features_class, feature_extraction(line))

    return np.reshape(all_features_class, (1, num_lines * num_features))


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

for i in range(1, num_classes + 1):
    print(i)
    training_dict[i] = np.genfromtxt('training' + str(i) + '.csv', delimiter=",")
    testing_dict[i] = np.genfromtxt('test' + str(i) + '.csv', delimiter=",")
start()
