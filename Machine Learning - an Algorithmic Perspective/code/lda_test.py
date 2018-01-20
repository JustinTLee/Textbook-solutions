# History:
# 2018-01-20 - JL - wrote test cases for meandev method and extract
#                   covariance matrix methods and LDA class

import numpy as np
import lda2
import unittest

class TestLDAMethods(unittest.TestCase):

    def setUp(self):
        # assertion comes from Linear Algebra and Its
        # Applications, David C. Lay, p. 485
        a0 = [[1, 2, 1], [4, 2, 13], [7, 8, 1], [8, 4, 5]]
        self.a0 = np.matrix(a0)
        self.b0 = [[-4, -2, -4], [-1, -2, 8], [2, 4, -4], [3, 0, 0]]

        # Found as test code in Machine Learning, an Algorithmic Perspective,
        # 2nd ed, Stephen Marsland, Ch.6 code: lda.py
        data1 = [[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.35,0.3],[0.4,0.4],[0.6,0.4],[0.7,0.45],[0.75,0.4],[0.8,0.35]]
        data1 = np.matrix(data1)
        labels1 = [0,0,0,0,0,1,1,1,1]
        self.a1SW, self.a1SB = lda2.calculateCovMat(data1, labels1)
        self.a1SW = self.a1SW.tolist()
        self.a1SB = self.a1SB.tolist()
        self.b1SW = [[ 0.079875, 0.049], [0.049, 0.057]]
        self.b1SB = [[ 0.435125, 0.13766667], [ 0.13766667, 0.04355556]]

        data2 = [[1, 1], [2, 2], [9, 9], [10, 10]]
        data2 = np.matrix(data2)
        labels2 = [0, 0, 1, 1]
        self.a2SW, self.a2SB = lda2.calculateCovMat(data2, labels2)
        self.a2SW = self.a2SW.tolist()
        self.a2SB = self.a2SB.tolist()
        self.b2SW = [[1, 1], [1, 1]]
        self.b2SB = [[64, 64], [64, 64]]

        test1 = lda2.lda()
        self.newData1trained = test1.trainWeights(data1, labels1, 2)
        self.weightstrained = test1.weights;
        self.newData1actual = [[-0.20799281, -0.10126718], [-0.17082239, -0.0360899 ], [-0.13365197, 0.02908738], [-0.09024709, 0.01400504], [-0.09648156, 0.09426466], [0.07713797, 0.03393528], [ 0.13912806, 0.05144157], [0.20735262, -0.01131176], [0.27557717, -0.07406509]]
        self.weightsactual = [[0.86809764, -0.30164691], [-0.49639348, 0.95341971]]

    def test_meandev_method_1(self):
        self.assertListEqual(lda2.meandev(self.a0).tolist(), self.b0)

    def test_extractSW_method_1(self):
        # Found as test code in Machine Learning, an Algorithmic Perspective,
        # 2nd ed, Stephen Marsland, Ch.6 code: lda.py
        np.testing.assert_almost_equal(self.a1SW, self.b1SW)
        
    def test_extractSW_method_2(self):
        np.testing.assert_almost_equal(self.a2SW, self.b2SW)
    
    def test_extractSB_method_1(self):
        np.testing.assert_almost_equal(self.a1SB, self.b1SB)

    def test_extractSB_method_2(self):
        np.testing.assert_almost_equal(self.a2SB, self.b2SB)

    def test_LDA_class(self):
        np.testing.assert_almost_equal(self.newData1trained, self.newData1actual)
        np.testing.assert_almost_equal(self.weightstrained, self.weightsactual)
        
if __name__ == '__main__':
    unittest.main(verbosity = 2)