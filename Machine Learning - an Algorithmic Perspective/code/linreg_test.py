import numpy as np
import linreg
import unittest

# History:
# 2018-01-21 - JL - wrote test case for: - instantiating method
#										 - comparing output
#										 - new output

class TestLinRegMethods(unittest.TestCase):

	def setUp(self):
		self.featMat = np.matrix([1, 2, 3, 4]).T
		self.y = np.array([3, 5, 7, 9]).T

		self.test1 = linreg.linreg("normal")
		self.y_hat = self.test1.calcBeta(self.featMat, self.y)

	def test_method_1(self):
		self.assertEqual(self.test1.method, "normal")

	def test_method_2(self):
		test2 = linreg.linreg()
		self.assertEqual(test2.method, "normal")

	def test_output(self):
		np.testing.assert_almost_equal(self.y_hat, np.reshape(self.y, (np.shape(self.y)[0], 1)))

	def test_newdata(self):
		featMat2 = np.matrix([11, 12, 13, 14]).T
		y2 = np.array([23, 25, 27, 29]).T

		y_hat2 = self.test1.calcMeanReg(featMat2)
		np.testing.assert_almost_equal(y_hat2, np.reshape(y2, (np.shape(y2)[0], 1)))

if __name__ == '__main__':
    unittest.main(verbosity = 2)