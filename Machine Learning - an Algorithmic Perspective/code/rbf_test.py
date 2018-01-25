import numpy as np
import rbf
import unittest

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
        self.rbf_test1 = rbf.rbf(1, 20, eta = 0.25, nIter = 100)

    def test_eta_init(self):
        self.assertEqual(self.rbf_test1.eta, 0.25)

    def test_iter_init(self):
        self.assertEqual(self.rbf_test1.iter, 100)

if __name__ == '__main__':
    unittest.main(verbosity = 2)