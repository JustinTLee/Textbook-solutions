import numpy as np
import datetime as dt
import pcn
import unittest

# History:
# 2018-01-20 - JL - wrote test case for: - initializing nWeights, iter, seed, and thresh_type attributes
#										 - creating weight matrix with proper dimension
#										 - predicting properly

class TestPCNClass(unittest.TestCase):

	def setUp(self):
		self.test_pcn1 = pcn.pcn(1, 20180120, thresh_type = 1)
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
		self.assertEqual(self.test_pcn1.thresh_type, 1)

	def test_thresh_type_init_2(self):
		self.assertEqual(self.test_pcn2.thresh_type, 1)

	def test_weights_dim(self):
		self.assertEqual(np.shape(self.test_pcn1.weights)[0], 3)
		self.assertEqual(np.shape(self.test_pcn2.weights)[1], 1)

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

if __name__ == '__main__':
	unittest.main(verbosity = 2)