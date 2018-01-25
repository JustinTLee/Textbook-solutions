import numpy as np
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
        distance = np.dot(data, self.weights) ** 2
        yi = math.exp(-distance/(2*self.sigma ** 2))

        return yi

test = rbfneuron(1, 30);


class rbf:

    def __init__(self, nNeurons, sigma, seed = None, eta = 0.25, nIter = 100):

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

        def trainWeights(self, data, labels):
            return


    def initializeNeurons(self, nFeatVar):
        # create vector of neurons
        self.matRBFNeurons = [rbfneuron(self.nNeurons, seed = self.seed + k, thresh_type = self.thresh_type) for k in range(self.nNeurons)]
        self.matNeurons = [pcn.neuron(nFeatVar + 1, seed = self.seed + k, thresh_type = self.thresh_type) for k in range(nFeatVars)]