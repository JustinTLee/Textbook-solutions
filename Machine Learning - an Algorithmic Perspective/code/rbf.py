import numpy as np
import datetime as dt
import pcn
import math

# History:
# 2018-01-24 - JL - started RBF class definition
# 2018-01-25 - JL - started RBF neuron class definition

class rbfneuron(pcn.neuron):

    def __init__(self, nInputs, sigma, seed = None):
        pcn.neuron.__init__(self, nInputs, seed)

        # set standard deviation
        self.sigma = sigma

        # overwrite unused attributes
        self.seed = None
        self.weights = None
        self.momentum = None
        self.thresh_type = None

    def thresholdH():
        pass

    def predictLabels(self, data):
        distance = np.asarray(np.dot(data, self.weights.T)) ** 2
        yi = np.exp(-distance/(2*self.sigma ** 2))

        return yi

class rbf:

    def __init__(self, nNeurons, sigma, seed = None, eta = 0.25, nIter = 100, thresh_type = 'logistic'):

        #initialize number of RBF kernels
        self.nNeurons = nNeurons

        # initialize standard devation of RBFs
        self.sigma = sigma

        # initialize learning rate
        self.eta = eta

        # set seed to be the date class was instantiated if no seed provided
        if seed is None:
            dtNow = dt.datetime.now()
            numNow = int(str(dtNow.year) + str(dtNow.month).zfill(2) + str(dtNow.day).zfill(2))
            self.seed = numNow
        else:
            self.seed = seed

        # set seed for class
        np.random.seed(self.seed)

        # initialize number of iterations to train
        self.iter = nIter

        # initialize threshold
        self.thresh_type = thresh_type

    def initializeNeurons(self, data):
        # get input data attributes
        nData = np.shape(data)[0]

        vectReorder = np.arange(nData)
        np.random.shuffle(vectReorder)
        dataReordered = data[vectReorder, :]

        # create vector of RBFneurons
        self.matRBFNeurons = [rbfneuron(nData, self.sigma) for k in range(self.nNeurons)]

        for m in range(self.nNeurons):
            self.matRBFNeurons[m].weights = dataReordered[m, :]

        # Create PCN layer
        self.PCNLayer = pcn.pcn(self.outputDim, seed = self.seed, thresh_type = self.thresh_type)

        # Create vector of PCN neurons
        self.PCNLayer.initializeNeurons(self.nNeurons)

    def forwardPredict(self, data, internal_bool = False):
        # create intermediate dataset
        h = np.zeros((np.shape(data)[0], self.nNeurons))

        # create predicted labels from data using every neuron
        y = np.zeros((np.shape(data)[0], self.outputDim))
        print(y)

        for m in range(self.nNeurons):
            h[:, m] = np.squeeze(self.matRBFNeurons[m].predictLabels(data))

        # add bias node
        bias = -1*np.ones((np.shape(data)[0], 1)) # bias weight
        data = np.concatenate((bias, h), axis = 1)

        for n in range(self.PCNLayer.nNeurons):
            yi = self.PCNLayer.matNeurons[n].predictLabels(data)

            y[:, n] = np.squeeze(yi)

        if internal_bool == True:
            return y
        elif internal_bool == False and self.thresh_type == 'logistic':
            y = np.where(y >= 0.5, 1, 0)
            return y
        else:
            return y

    def trainWeights(self, data, labels):
        # get input data attributes
        nData = np.shape(data)[0]
        nFeatVar = np.shape(data)[1]

        # transpose labels to fit feature matrix layout
        labels = np.matrix(labels)
        if np.shape(labels)[0] < np.shape(labels)[1]:
            labels = labels.T

        # get label dimension in order to initialize matrix for predicted labels
        self.outputDim = np.shape(labels)[1]

        self.initializeNeurons(data)

        # train weights
        for m in range(self.iter):
            vectReorder = np.arange(nData)
            np.random.shuffle(vectReorder)
            dataReordered = data[vectReorder, :]
            labelsReordered = labels[vectReorder, :]


data = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1])

test = rbf(4, 20, seed = 56)
test.trainWeights(data, labels)
print(test.forwardPredict(data))