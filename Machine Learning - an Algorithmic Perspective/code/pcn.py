import numpy as np
import datetime as dt

# History:
# 2018-01-20 - JL - started pcn class, created instantiation
#                 - created thresholding function (gating)
#                 - created label prediction function
#                 - created training function
# 2018-01-21 - JL - made changes to label reshaping to account for multidimensional labels
#                 - made neuron its own class and modified perceptron to loop through amount of neurons
#                 - add ability to concatenate intercept feature variable in PCN class if not present

class neuron:
# neuron can: - perform dot product on weights and inputs
#             - threshold intermediate to get output
#             - change thresholding scheme
#             - initialize weights 

    def __init__(self, nInputs, seed = None, thresh_type = 'step'):
        self.nInputs = nInputs

        # set seed to be the date class was instantiated if no seed provided
        if seed is None:
            dtNow = dt.datetime.now()
            numNow = int(str(dtNow.year) + str(dtNow.month).zfill(2) + str(dtNow.day).zfill(2))
            self.seed = numNow
        else:
            self.seed = seed

        # initialize threshold
        if thresh_type != 'step':
            self.thresh_type = thresh_type
        else:
            self.thresh_type = 'step'

        # set seed for class
        np.random.seed(self.seed)

        # initialize weights
        self.weights = np.random.uniform(-1, 1, (self.nInputs, 1))

    def thresholdH(self, hij):
        # use various activation functions to threshold result

        # simple binary threshold to see if hij is above or below 0
        if self.thresh_type == 'step':
            yi = np.where(hij > 0, 1, 0)

        # logistic threshold with boundary at 0.5
        elif self.thresh_type == 'logistic':
            yi = 1/(1 + np.exp(-hij))

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

    def __init__(self, nNeurons, seed = None, iter = None, thresh_type = 'step'):
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
        if thresh_type != 'step':
            self.thresh_type = thresh_type
        else:
            self.thresh_type = 'step'
        
        # set seed for class
        np.random.seed(self.seed)

        # generate eta as random number between 0.1 to 0.4, as stated in p. 46
        self.eta = np.random.uniform(0.1, 0.4, 1)

    def forwardPredict(self, data, internal_bool = False):
        # create predicted labels from data using every neuron
        y = np.zeros((np.shape(data)[0], self.outputDim))

        # add bias node
        if ~np.all(data[:, 0] == -1):
            bias = -1*np.ones((np.shape(data)[0], 1)) # bias weight
            data = np.concatenate((bias, data), axis = 1)

        for n in range(self.nNeurons):
            yi = self.matNeurons[n].predictLabels(data)

            y[:, n] = np.squeeze(yi)

        if internal_bool == True:
            return y
        elif internal_bool == False and self.thresh_type == 'logistic':
            y = np.where(y >= 0.5, 1, 0)
            return y
        else:
            return y

    def initializeNeurons(self, nFeatVar):
        # create vector of neurons
        self.matNeurons = [neuron(nFeatVar + 1, seed = self.seed + k, thresh_type = self.thresh_type) for k in range(self.nNeurons)]

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

        self.initializeNeurons(nFeatVar)

        # train weights
        for m in range(self.iter):

            for n in range(self.nNeurons):
                # loop through every neuron
                # create label predictions
                yi = self.matNeurons[n].predictLabels(data)

                # update the weights
                self.matNeurons[n].weights -= np.squeeze(self.eta)*data.T*(yi - labels[:, n])

        y = self.forwardPredict(data)

        if self.thresh_type == 'logistic':
            y = np.where(y >= 0.5, 1, 0)
            return y
        else:
            return y

class mlp:
# multi-layer perceptron can: - initialize number of layers
#                             - run forward algorithm to calculate predicted labels
#                             - calculate weights for each neuron in MLP

    def __init__(self, matLayers, seed = None, iter = None, thresh_type = 'step'):
        self.nLayers = np.shape(matLayers)[0]

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
        if thresh_type != 'step':
            self.thresh_type = thresh_type
        else:
            self.thresh_type = 'step'

        # set seed for class
        np.random.seed(self.seed)

        self.matPCN = [pcn(matLayers[k], seed = self.seed + k, iter = self.iter, thresh_type = self.thresh_type) for k in range(self.nLayers)]

    def forwardPredict(self, data, internal_bool = False):
        data_int = data
        output = []
        output.append(data_int)

        for k in range(self.nLayers):
            data_int = self.matPCN[k].forwardPredict(data_int, internal_bool = True)
            output.append(data_int)

        if internal_bool == True:
            return output
        else:
            if self.thresh_type == 'logistic':
                output = np.where(output[self.nLayers] >= 0.5, 1, 0)
                return output
            else:
                return output[self.nLayers]

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

        # add bias node
        bias = -1*np.ones((nData, 1)) # bias weight
        data = np.concatenate((bias, data), axis = 1)

        # initialize neurons
        for k in range(self.nLayers):

            # initialize the number of neurons in the output layer to be the target dimensions
            if k == (self.nLayers - 1):
                if np.shape(labels)[1] > self.matPCN[k].nNeurons:
                    self.matPCN[k].nNeurons = np.shape(labels)[1]
                self.matPCN[k].initializeNeurons(self.matPCN[k - 1].nNeurons)

            # initialize the number of weights in the first layer to be equal to the amount of feature variables
            if k == 0:
                self.matPCN[k].initializeNeurons(nFeatVar)

            # initialize the number of neurons in middle hidden layers to be the amount of inputs from the previous layer
            else:
                self.matPCN[k].initializeNeurons(self.matPCN[k - 1].nNeurons)

            if k == (self.nLayers - 1):
                self.matPCN[k].outputDim = np.shape(labels)[1]
            else:
                self.matPCN[k].outputDim = self.matPCN[k].nNeurons

        # train weights
        for m in range(self.iter):
            y = self.forwardPredict(data, internal_bool = True)
            print(y)

            for n in range(self.nLayers, 0, -1):
                
                layer_str = "\nLayer: " + str(n)
                print(layer_str)
                neuron_str = "Number of neurons in layer #" + str(n) + ": " + str(self.matPCN[n - 1].nNeurons)
                print(neuron_str)

                if n == self.nLayers:

                    ErrorNO = np.zeros(np.shape(labels))

                    for o in range(self.matPCN[n - 1].nNeurons):
                        weights_str = "Weights of neuron #" + str(o + 1) + " in layer #" + str(n) + " :\n" + str(self.matPCN[n - 1].matNeurons[o].weights) + "\n"
                        print(weights_str)

                        outputNO = np.matrix(y[n][:, o]).T
                        ErrorNO[:, o] = np.squeeze((outputNO - labels[:, o])*outputNO.T*(1 - outputNO))
                        previousLayerOutput = np.matrix(y[n - 1])

                        # add bias node
                        if ~np.all(previousLayerOutput[:, 0] == -1):
                            onesVect = -1*np.ones((np.shape(previousLayerOutput)[0], 1))
                            previousLayerOutput = np.concatenate((onesVect, previousLayerOutput), axis = 1)

                        self.matPCN[n - 1].matNeurons[o].weights -= float(self.matPCN[n - 1].eta)*(previousLayerOutput.T*np.matrix(ErrorNO[:, o]).T)

                    print(ErrorNO)

                else:

                    for o in range(self.matPCN[n - 1].nNeurons):
                        weights_str = "Weights of neuron #" + str(o + 1) + " in layer #" + str(n) + " :\n" + str(self.matPCN[n - 1].matNeurons[o].weights) + "\n"
                        print(weights_str)

                        layerNOOutput = np.matrix(y[n][:, o]).T
                        print(layerNOOutput)
                        for p in range(self.matPCN[n].nNeurons):
                            nextLayerErrorP = np.dot(ErrorNO, self.matPCN[n].matNeurons[p].weights.T)
                            if p == 0:
                                nextLayerError = nextLayerErrorP
                            else:
                                nextLayerError = nextLayerError + nextLayerErrorP

                        print(nextLayerError)
                        layerErrorNO = layerNOOutput*(1 - layerNOOutput).T*nextLayerError
                        print(layerErrorNO)
                        print(self.matPCN[n - 1].matNeurons[0].weights)
                        self.matPCN[n - 1].matNeurons[0].weights -= float(self.matPCN[n - 1].eta)*(layerNOOutput.T*layerErrorNO).T

        y = self.forwardPredict(data)

        return y

# test = pcn(2, iter = 100, thresh_type = "logistic")

# data = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
# data2 = np.matrix([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [1, 1]])
# target = np.matrix([[1, 1, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]])
# output = test.trainWeights(data, target)
# print(test.thresh_type)
# print(output)
# print(test.forwardPredict(data2))

# test = mlp([1, 1, 1], iter = 1, thresh_type = 'logistic')

# data = np.matrix([[1, 1], [1, 1], [1, 1], [1, 1]])
# target = np.matrix([[0, 0], [0, 0], [0, 0], [0, 0]])
# print(test.trainWeights(data, target))

test = mlp([1, 1, 1], iter = 100, thresh_type = 'logistic')
data = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.matrix([[0], [0], [0], [1]])
# target = np.matrix([[1, 1, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]])

print(test.trainWeights(data, target))