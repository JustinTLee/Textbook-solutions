import numpy as np
import rbf
import unittest

class TestRBFClass(unittest.TestCase):

    def setUp(self):
        self.rbf_test1 = rbf.rbf(eta = 0.25, nIter = 100)

    def test_eta_init(self):
        self.assertEqual(self.rbf_test1.eta, 0.25)

    def test_iter_init(self):
        self.assertEqual(self.rbf_test1.iter, 100)

if __name__ == '__main__':
    unittest.main(verbosity = 2)