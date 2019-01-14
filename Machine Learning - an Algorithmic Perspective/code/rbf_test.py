import numpy as np
import rbf
import unittest

# History:
# 2018-01-25 - JL - added test cases for RBFNeuron class and RBF class
# 2018-01-26 - JL - added methods for RBNeuron class:
#                                                     - test_thresh_type_init
#                                                     - test_outputDim_init
#                                                     - test_PCNNeurons_init
#                                                     - test_output_init

class TestRBFNeuronClass(unittest.TestCase):

    def setUp(self):
        self.neuron1 = rbf.rbfneuron(1, 30)

    def test_nInputs_init(self):
        self.assertEqual(self.neuron1.nInputs, 1)

    def test_std_init(self):
        self.assertEqual(self.neuron1.sigma, 30)

    def test_setNone_init(self):
        self.assertEqual(self.neuron1.seed, None)
        self.assertEqual(self.neuron1.momentum, None)
        self.assertEqual(self.neuron1.thresh_type, None)

    def test_predictLabels(self):
        datapoint1 = np.matrix([[0]])
        self.neuron1.weights = np.matrix([[0]])
        self.assertEqual(self.neuron1.predictLabels(datapoint1), 1)

class TestRBFClass(unittest.TestCase):

    def setUp(self):
        self.rbf_test1 = rbf.rbf(4, 0.5, eta = 0.25, nIter = 100, thresh_type = "logistic")
        self.data = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.labels = np.array([0, 1, 1, 0])
        self.rbf_test1.trainWeights(self.data, self.labels)

    def test_eta_init(self):
        self.assertEqual(self.rbf_test1.eta, 0.25)

    def test_iter_init(self):
        self.assertEqual(self.rbf_test1.iter, 100)

    def test_thresh_type_init(self):
        self.assertEqual(self.rbf_test1.thresh_type, "logistic")
        self.assertEqual(self.rbf_test1.PCNLayer.thresh_type, "logistic")
        self.assertEqual(self.rbf_test1.PCNLayer.matNeurons[0].thresh_type, "logistic")

    def test_outputDim_init(self):
        self.assertEqual(self.rbf_test1.outputDim, 1)

    def test_PCNNeurons_init(self):
        self.assertEqual(self.rbf_test1.PCNLayer.nNeurons, 1)
        self.assertEqual(np.shape(self.rbf_test1.PCNLayer.matNeurons[0].weights), (5, 1))

    def test_output_init(self):
        output = self.rbf_test1.forwardPredict(self.data)
        output = np.squeeze(np.where(output > 0.5, 1, 0))
        np.testing.assert_equal(output, self.labels)

if __name__ == '__main__':
    unittest.main(verbosity = 2)