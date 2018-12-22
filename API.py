from flask import Flask  , request,send_from_directory
import time
import cv2
import AdjustRotation as AR
from fast_test import *
from segmentation import *
from AnglesHistogram import *
from BlobsDetection import *
from ConnectedComponents import *
from DiskFractal import *
import glob
from sklearn import neighbors
import warnings
from itertools import combinations
import random


warnings.filterwarnings("ignore", category=RuntimeWarning)
testing_dict = {}
for i in range(1, 9):
    print(i)
    training_dict[i] = np.genfromtxt('database/training' + str(i) + '.csv', delimiter=",")
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# @app.route('/')
# def hello_world():
#     return 'Hello, World!'

@app.route('/', methods=['GET', 'POST'])
def import_image():
    """Import the data for this recipe by either saving the image associated
    with this recipe or saving the metadata associated with the recipe. If
    the metadata is being processed, the title and description of the recipe
    must always be specified."""
    try:
        if 'captured_image' in request.files:
            images = request.files['captured_image']
            filename = 'rotated'+ str(int(time.time()))+'.jpg'
            images.save(UPLOAD_FOLDER+filename)
            rotated = cv2.imread(UPLOAD_FOLDER+filename)
            rotated = AR.adjust_rotation(rotated)
            cv2.imwrite(UPLOAD_FOLDER+filename,rotated)
    except KeyError as e:
        return 'error',404
    return filename

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER,filename,as_attachment=True)

@app.route('/identify',methods=['GET', 'POST'])
def identification():
    test_combination1 = (request.form['class1'])
    test_combination3 = (request.form['class3'])
    test_combination2 = (request.form['class2'])
    filename = UPLOAD_FOLDER+(request.form['filename'])
    testing_image = cv2.imread(filename)
    test_vector = test(testing_image)
    test_combination = [int(test_combination1),int(test_combination2),int(test_combination3)]
    num_features = 18
    classifier = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    all_features = np.asarray([])
    labels = np.asarray([])
    num_training_examples = 0

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



    test_vector = (test_vector - mu) / sigma
    prediction = classifier.predict(test_vector.reshape(1, -1))
    # print(prediction)

    return str(int(prediction[0])),200

app.run()