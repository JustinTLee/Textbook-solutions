import numpy as np
import datetime as dt

# History:
# 2018-01-20 - JL - started pcn class, created instantiation
#                 - created thresholding function (gating)
#                 - created label prediction function
#                 - created training function
# 2018-01-21 - JL - made changes to label reshaping to account for multidimensional labels
#                 - made neuron its own class and modified perceptron to loop through amount of neurons

class neuron:
# neuron can: - perform dot product on weights and inputs
#             - threshold intermediate to get output
#             - change thresholding scheme
#             - initialize weights 

    def __init__(self, nInputs, seed = None, thresh_type = 'linear'):
        self.nInputs = nInputs

        # set seed to be the date class was instantiated if no seed provided
        if seed is None:
            dtNow = dt.datetime.now()
            numNow = int(str(dtNow.year) + str(dtNow.month).zfill(2) + str(dtNow.day).zfill(2))
            self.seed = numNow
        else:
            self.seed = seed

        # initialize threshold
        if thresh_type != 'linear':
            self.thresh_type = thresh_type
        else:
            self.thresh_type = 'linear'

        # set seed for class
        np.random.seed(self.seed)

        # initialize weights
        self.weights = np.random.uniform(-1, 1, (self.nInputs, 1))

    def thresholdH(self, hij):
        # use various activation functions to threshold result

        # simple binary threshold to see if hij is above or below 0
        if self.thresh_type == 'linear':
            yi = np.where(hij > 0, 1, 0)

        # logistic threshold with boundary at 0.5
        elif self.thresh_type == 'logistic':
            logit_ij = 1/(1 + np.exp(-hij))
            yi = np.where(logit_ij > 0.5, 1, 0)

        # if any-non empty string other than the possible options, return zero array
        else:
            yi = np.zeros((np.shape(hij)[0], 1))

        return yi

    def predictLabels(self, data):
        # multiply input by weights to get hij
        hij = np.dot(data, self.weights)

        # threshold hij according to some function
        yi = self.thresholdH(hij)

        return yi

class pcn:
# perceptron can: - initialize number of neurons
#                 - update neuron weights
#                 - run forward algorithm to calculate predicted labels

    def __init__(self, nNeurons, seed = None, iter = None, thresh_type = 'linear'):
        # Instatiate class with number of weights, seed, and eta
        self.nNeurons = nNeurons

        # set seed to be the date class was instantiated if no seed provided
        if seed is None:
            dtNow = dt.datetime.now()
            numNow = int(str(dtNow.year) + str(dtNow.month).zfill(2) + str(dtNow.day).zfill(2))
            self.seed = numNow
        else:
            self.seed = seed

        # initialize number of iterations
        if iter is None:
            self.iter = 20
        else:
            self.iter = iter

        # initialize threshold
        if thresh_type != 'linear':
            self.thresh_type = thresh_type
        else:
            self.thresh_type = 'linear'
        
        # set seed for class
        np.random.seed(self.seed)

        # generate eta as random number between 0.1 to 0.4, as stated in p. 46
        self.eta = np.random.uniform(0.1, 0.4, 1)

    def forwardPredict(self, data):
        # create predicted labels from data using every neuron
        y = np.zeros((np.shape(data)[0], self.outputDim))

        for n in range(self.nNeurons):
            yi = self.matNeurons[n].predictLabels(data)

            y[:, n] = np.squeeze(yi)

        return y

    def trainWeights(self, data, labels):
        # get input data attributes
        nData = np.shape(data)[0]
        nFeatVar = np.shape(data)[1]

        # transpose labels to fit feature matrix layout
        labels = np.matrix(labels)
        if np.shape(labels)[0] < np.shape(labels)[1]:
            labels = labels.T

        # Since each neuron represents a dimension of the output, if the target matrix has a higher
        # dimension than the number of neurons initiated, then make the number of neurons equal to
        # that dimension
        if np.shape(labels)[1] > self.nNeurons:
            self.nNeurons = np.shape(labels)[1]

        # get label dimension in order to initialize matrix for predicted labels
        self.outputDim = np.shape(labels)[1]

        # add bias node
        bias = -1*np.ones((nData, 1)) # bias weight
        data = np.concatenate((bias, data), axis = 1)

        # create vector of neurons
        self.matNeurons = [neuron(nFeatVar + 1, seed = self.seed + k, thresh_type = self.thresh_type) for k in range(self.nNeurons)]

        for m in range(self.iter):

            for n in range(self.nNeurons):
                # loop through every neuron
                # create label predictions
                yi = self.matNeurons[n].predictLabels(data)

                # update the weights
                self.matNeurons[n].weights -= np.squeeze(self.eta)*data.T*(yi - labels[:, n])

        y = self.forwardPredict(data)

        return y