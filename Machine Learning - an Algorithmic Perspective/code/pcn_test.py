import numpy as np
import datetime as dt
import pcn
import unittest

# History:
# 2018-01-20 - JL - wrote test case for: - initializing nWeights, iter, seed, and thresh_type attributes
#                                        - creating weight matrix with proper dimension
#                                        - predicting properly
# 2018-01-21 - JL - split up class test cases between new neuron clsas and pcn class
#                 - create new test cases for neuron class
# 2018-01-22 - JL - added tests for new MLP class
# 2018-01-24 - JL - add soft-max test

class TestNeuronClass(unittest.TestCase):

    def setUp(self):
        self.test = pcn.neuron(3)
        self.data = np.matrix([[-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 1, 1]])
        self.target1 = np.matrix([[0], [0], [0], [1]])
        self.target2 = np.matrix([[0], [1], [1], [1]])

    def test_nInputs_init(self):
        self.assertEqual(self.test.nInputs, 3)

    def test_seed_init_1(self):
        dtNow = dt.datetime.now()
        numNow = int(str(dtNow.year) + str(dtNow.month).zfill(2) + str(dtNow.day).zfill(2))
        test2 = pcn.neuron(1)
        self.assertEqual(self.test.seed, numNow)

    def test_seed_init_2(self):
        dtNow = dt.datetime.now()
        numNow = int(str(dtNow.year) + str(dtNow.month).zfill(2) + str(dtNow.day).zfill(2))
        test2 = pcn.neuron(1)
        self.assertEqual(test2.seed, numNow)

    def test_thresh_type_init_1(self):
        self.assertEqual(self.test.thresh_type, 'step')

    def test_thresh_type_init_2(self):
        test2 = pcn.neuron(2, thresh_type = "logistic")
        self.assertEqual(test2.thresh_type, 'logistic')
        test2 = pcn.neuron(2, thresh_type = 'soft-max')
        self.assertEqual(test2.thresh_type, 'soft-max')

    def test_weights_dim_init(self):
        self.assertEqual(np.shape(self.test.weights), (3, 1))

    def test_thresholdH(self):
        self.assertEqual(self.test.thresholdH(0.5), 1)
        self.assertEqual(self.test.thresholdH(-0.5), 0)
        self.assertEqual(self.test.thresholdH(0), 0)

        # test logistic thresholding
        test2 = pcn.neuron(2, thresh_type = "logistic")
        self.assertEqual(test2.thresholdH(0.9) > 0.5, 1)
        self.assertEqual(test2.thresholdH(-0.3) > 0.5, 0)

    def test_predictLabels(self):
        self.assertEqual(np.shape(self.test.predictLabels(self.data)), (4, 1))

class TestPCNClass(unittest.TestCase):

    def setUp(self):
        self.test_pcn1 = pcn.pcn(1, 20180120, thresh_type = 'logistic')
        self.test_pcn2 = pcn.pcn(1, iter = 10)

        self.data = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.targets1 = np.array([0, 0, 0, 1])
        self.targets2 = np.array([0, 1, 1, 1])

        self.predict1 = np.squeeze(self.test_pcn1.trainWeights(self.data, self.targets1))
        self.predict2 = np.squeeze(self.test_pcn2.trainWeights(self.data, self.targets2))

    def test_nWeights_init(self):
        self.assertEqual(self.test_pcn1.nNeurons, 1)

    def test_seed_init_1(self):
        self.assertEqual(self.test_pcn1.seed, 20180120)

    def test_seed_init_2(self):
        dtNow = dt.datetime.now()
        numNow = int(str(dtNow.year) + str(dtNow.month).zfill(2) + str(dtNow.day).zfill(2))
        self.assertEqual(self.test_pcn2.seed, numNow)

    def test_iter_init_1(self):
        self.assertEqual(self.test_pcn1.iter, 20)

    def test_iter_init_2(self):
        self.assertEqual(self.test_pcn2.iter, 10)

    def test_thresh_type_init_1(self):
        self.assertEqual(self.test_pcn1.thresh_type, 'logistic')

    def test_thresh_type_init_2(self):
        self.assertEqual(self.test_pcn2.thresh_type, 'step')

    def test_initializeneurons(self):
        test_pcn3 = pcn.pcn(1)
        test_pcn3.initializeNeurons(np.shape(self.data)[1])
        self.assertEqual(np.shape(test_pcn3.matNeurons[0].weights), (3, 1))

    def test_weights_dim(self):
        self.assertEqual(np.shape(self.test_pcn1.matNeurons[0].weights), (3, 1))
        self.assertEqual(np.shape(self.test_pcn2.matNeurons[0].weights), (3, 1))

    def test_predict_1(self):
        np.testing.assert_equal(self.predict1.tolist(), self.targets1)

    def test_predict_2(self):
        np.testing.assert_equal(self.predict2.tolist(), self.targets2)

    def test_predict_3(self):
        data3D = np.matrix([[0, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        targets3 = np.array([0, 1, 1, 0])
        test_pcn3 = pcn.pcn(1, iter = 30)
        predict3 = np.squeeze(test_pcn3.trainWeights(data3D, targets3))
        np.testing.assert_equal(predict3.tolist(), targets3)

class TestMLPClass(unittest.TestCase):

    def setUp(self):
        self.test_mlp = pcn.mlp([2, 2, 1], seed = 20180122, iter = 200)
        self.data = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.nFeatvar = np.shape(self.data)[1]

    def test_layers_init(self):
        self.assertEqual(self.test_mlp.nLayers, 3)

    def test_nNeurons_init(self):
        self.assertEqual(self.test_mlp.matPCN[0].nNeurons, 2)
        self.assertEqual(self.test_mlp.matPCN[1].nNeurons, 2)
        self.assertEqual(self.test_mlp.matPCN[2].nNeurons, 1)

    def test_predict1(self):
        test_mlp2 = pcn.mlp([20, 1], iter = 8000)
        targets = np.array([[0], [1], [1], [0]])
        output = test_mlp2.trainWeights(self.data, targets)
        np.testing.assert_equal(output.tolist(), targets)


if __name__ == '__main__':
    unittest.main(verbosity = 2)