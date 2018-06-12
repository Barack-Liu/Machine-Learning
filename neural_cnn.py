from sys import argv
import sys

sys.path.append('neural-networks-and-deep-learning/src')
from os import listdir
from os.path import isdir, isfile, join
import network3
import numpy as np
import cv2
import random
import pickle
from matplotlib import pyplot as plt
from utils import getInput, showImage, printErrorMsg

TOTAL_SIZE = 784
SIDE_SIZE = 28
N_ELEMENTS = 37  # [0-9][A-Z]ESP

"""
def getNeuralNetFromUser():
    neural_net_file = "resources/neural_net"  # JSON in a text file used to load the neural network
    net = None
    print "Load Neural Network from file?"
    value = getInput("-1 for training a new network, other key to load a trained one: ")
    if (value == '-1'):
        net_layers = [TOTAL_SIZE]  # List of neurons, input layer == N pixels
        i = 1
        print "For each layer, insert the number of neurons\nInsert -1 to finish: "
        while (True):
            s_layer = "Layer {}: ".format(i)
            layer = int(getInput(s_layer))
            if (layer == -1):
                break
            net_layers.append(layer)
            i += 1
        net_layers.append(N_ELEMENTS)  # Output layer == N possible output values
        net = network2.Network(net_layers, cost=network2.CrossEntropyCost)
        net.large_weight_initializer()
    else:
        value = getInput(
            "-1 for specifying the neural network file. Other to load the default '{}': ".format(neural_net_file))
        if (value == '-1'):
            neural_net_file = getInput("Insert file name of the neural net to be loaded: ")
            while (True):
                if (isfile(neural_net_file)):
                    break
                neural_net_file = getInput("Insert file name of the neural net to be loaded: ")
                print "Name invalid, please try again"
        net = network2.load(neural_net_file)  # Loads an existing neural network from a file
    return net
"""


def getCnnNetFromUser(mini_batch_size):
    neural_net_file = "cnn_plate.obj"
    net = None
    print "Load Neural Network from file?"
    value = getInput("-1 for training a new network, other key to load a trained one: ")
    if (value == '-1'):
        """
        net = network3.Network([
            network3.ConvPoolLayer(image_shape=(mini_batch_size, 1, SIDE_SIZE, SIDE_SIZE),
                                   filter_shape=(20, 1, 5, 5),
                                   poolsize=(2, 2),
                                   activation_fn=network3.ReLU),
            network3.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                                   filter_shape=(40, 20, 5, 5),
                                   poolsize=(2, 2),
                                   activation_fn=network3.ReLU),
            network3.FullyConnectedLayer(
                n_in=40 * 4 * 4, n_out=1000, activation_fn=network3.ReLU, p_dropout=0.5),
            network3.FullyConnectedLayer(
                n_in=1000, n_out=100, activation_fn=network3.ReLU, p_dropout=0.5),
            network3.SoftmaxLayer(n_in=100, n_out=37, p_dropout=0.5)],
            mini_batch_size)
        """
        net = network3.Network([
            network3.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                                   filter_shape=(20, 1, 5, 5),
                                   poolsize=(2, 2),
                                   activation_fn=network3.ReLU),
            network3.FullyConnectedLayer(
                n_in=20 * 12 * 12, n_out=100, activation_fn=network3.ReLU, p_dropout=0.5),
            network3.SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)],
            mini_batch_size)

    else:
        value = getInput(
            "-1 for specifying the neural network file. Other to load the default '{}': ".format(neural_net_file))
        if (value == '-1'):
            neural_net_file = getInput("Insert file name of the neural net to be loaded: ")
            while (True):
                if (isfile(neural_net_file)):
                    break
                neural_net_file = getInput("Insert file name of the neural net to be loaded: ")
                print "Name invalid, please try again"
        file_handler = open(neural_net_file, 'r')
        net = pickle.load(file_handler)
    return net


def processUserNeuralSettings():
    epochs = int(getInput("epochs: "))
    mini_batch_size = int(getInput("batch-size: "))
    eta = float(getInput("learning rate (< 1): "))
    lmbda = float(getInput("lambda: "))

    return epochs, mini_batch_size, eta, lmbda


def expandTrainingData(training_x, training_y):
    """ Takes every image on the training_data set and generates 4 additional images
		by displacing each one pixel up, down, left and right

		Note: the expanded_training_pairs contains the original training_data set
	"""
    expanded_training_x = []
    expanded_training_y = []
    C = 1  # /float(1)-0.1
    j = 0  # counter
    for (x, y) in zip(training_x, training_y):
        # expanded_training_pairs.append((np.reshape(x, (TOTAL_SIZE, 1)), y))
        expanded_training_x.append(x)
        expanded_training_y.append(y)
        image = np.reshape(x, (-1, SIDE_SIZE))

        j += 1
        if j % 1000 == 0:
            print("Expanding image number", j)
        # iterate over data telling us the details of how to
        # do the displacement
        for d, axis, index_position, index in [
            (1, 0, "first", 0),
            (-1, 0, "first", SIDE_SIZE - 1),
            (1, 1, "last", 0),
            (-1, 1, "last", SIDE_SIZE - 1)]:
            new_img = np.roll(image, d, axis)
            p = np.empty(SIDE_SIZE)
            p.fill(C)
            if index_position == "first":
                p = np.empty(SIDE_SIZE)
                p.fill(C)
                new_img[index, :] = p
            else:
                new_img[:, index] = p
            img = np.reshape(new_img, (TOTAL_SIZE, ))
            expanded_training_x.append(img)
            expanded_training_y.append(y)
    return expanded_training_x, expanded_training_y


def processImage(arg1, image=False):
    """ Takes an image path or a preprocessed image (if image=True) and converts it to binary threshold
		and reshapes it into a TOTAL_SIZE-D x 1 vector
	"""
    img = arg1
    if (not image):
        img = cv2.imread(arg1, 0)
        img = cv2.resize(img, (SIDE_SIZE, SIDE_SIZE), interpolation=cv2.INTER_CUBIC)
        img = cv2.normalize(img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # img = list(map((lambda x: x/float(1)-0.1), img))
    return np.reshape(img, (TOTAL_SIZE,))


def loadImgs(trainingPath, validatingPath=None, testingPath=None):
    """ Returns a list of tuples representing the image and the value necessary for feeding the neural network
		* If vPath is specified, returns its list of tuples too.
	"""

    def translateASCIIvalue(c):
        """ Returns the int position of a char or the string 'ESP' in a 37-D vector """
        if (c == 'ESP'):
            return 36
        v = ord(c)
        v = v - 55 if (v > 57) else v - 48  # Based on ASCII values
        return v

    def vectorizeValue(c):
        """ Returns a 37-D Vector with a value of 1.0 in the position associated to the char 'c' """
        result = np.zeros((N_ELEMENTS, 1))
        v = translateASCIIvalue(c)
        result[v] = 1.0
        return result

    def loadImgsAux(path, func):
        """ Processes all images and return a list of tuples (img --> TOTAL_SIZE-D x 1, value --> return of 'func') """
        imgs = []
        vs = []
        i = 0
        # Load images
        for f in listdir(path):
            if (isfile(join(path, f)) and f.endswith('.jpg')):
                i += 1
                if i % 1000 == 0: print("Adding image number", i)
                img = processImage(join(path, f))
                c = f.split('_')[0]
                v = func(c)
                imgs.append(img)
                vs.append(v)
        return np.array(imgs, dtype='f'), vs

    # Load training images
    # Tuples (img --> TOTAL_SIZE-D x 1, value --> int)
    trainingImg, trainingLabel = loadImgsAux(trainingPath, translateASCIIvalue)
    if validatingPath is None and testingPath is None:
        return trainingImg

    validatingImg, validatingLabel = None, None
    testingImg, testingLabel = None, None

    # Load testing images
    # Tuples (img --> TOTAL_SIZE-D x 1, value --> int)
    if validatingPath is not None and testingPath is None:
        validatingImg, validatingLabel = loadImgsAux(validatingPath, translateASCIIvalue)
        return trainingImg, trainingLabel, validatingImg, validatingLabel

    # Tuples (img --> TOTAL_SIZE-D x 1, value --> int)
    validatingImg, validatingLabel = loadImgsAux(validatingPath, translateASCIIvalue)

    testingImg, testingLabel = loadImgsAux(testingPath, translateASCIIvalue)
    return trainingImg, trainingLabel, validatingImg, validatingLabel, testingImg, testingLabel


######################################
#			    MAIN			 	 #
######################################

####### Param specification #######
def main():
    USE = "Use: <Script Name> <Training Dir> [Validating Dir] [Testing Dir]"
    if len(argv) < 2:
        printErrorMsg("Param number incorrect\n" + USE)
        exit(1)
    trainingPath = argv[1]
    validatingPath = None
    testingPath = None
    if not isdir(trainingPath):
        printErrorMsg("'" + trainingPath + "'" + " is not a valid directory\n" + USE)
        exit(1)
    if argv[2]:
        if not isdir(argv[2]):
            printErrorMsg("'" + argv[2] + "'" + " is not a valid directory\n" + USE)
            exit(1)
        validatingPath = argv[2]
    if argv[3]:
        if not isdir(argv[3]):
            printErrorMsg("'" + argv[3] + "'" + " is not a valid directory\n" + USE)
            exit(1)
        testingPath = argv[3]

    training_x, training_y, validation_x, validation_y, test_x, test_y = loadImgs(trainingPath, validatingPath,
                                                                                  testingPath)
    value = getInput("-1 to expand training data, skip otherwise: ")
    if value == '-1':
        training_x, training_y = expandTrainingData(training_x, training_y)

    epochs = 40
    mini_batch_size = 10
    eta = 0.5
    lmbda = 0
    print "\nDefinition of the neural network"
    print "################################"
    value = getInput("-1 to manually specify parameters, other key for default: ")
    if value == '-1':
        epochs, mini_batch_size, eta, lmbda = processUserNeuralSettings()

    # Allows to load an existing network or creating a new network
    net = getCnnNetFromUser(mini_batch_size)
    # Load the Plate data to Shared Memory
    training_data, validating_data, testing_data = network3.load_plateData_shared(
        training_x, training_y, validation_x, validation_y, test_x, test_y)

    value = getInput("0 to test the accuracy and cost of the trained network, other key for training: ")
    if value == '0':
        accuracy = net.accuracy(training_data)
        print "Accuracy on training data: {:.2%}".format(accuracy)

        accuracy = net.accuracy(validating_data)
        print "Accuracy on validation data: {:.2%}".format(accuracy)

        accuracy = net.accuracy(testing_data)
        print "Accuracy on test data: {:.2%}".format(accuracy)

        """
        prediction = net.predict(test_data)
        print "Prediction of test_data: {}".format(prediction)
        """

    while (True):
        #	print "TRAINGING NEURAL NET?"
        #	value = getInput("-1 for exit, other key for training: ")
        if value == '0':
            break
        print "TRAINING NEURAL NET"
        print "###################"
        net.SGD(training_data, epochs, mini_batch_size, eta,
                validating_data, testing_data)
        print "Save neural net?"
        value = getInput("-1 for not saving, other key for saving: ")
        if value != '-1':
            neural_net_file = getInput("Name: ")
            file_handler = open(neural_net_file, 'w')
            pickle.dump(net, file_handler)
        print "Keep training?"
        value = getInput("-1 for exit, other key for trying: ")
        if value == '-1':
            break
        value = getInput("-1 to manually specify parameters, other key for default: ")
        if value == '-1':
            epochs, mini_batch_size, eta, lmbda = processUserNeuralSettings()
        value = 0

    exit(0)


if __name__ == "__main__":
    main()
