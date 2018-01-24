import numpy as numpy
import pcn
import math

class rbf:

	def __init__(self, eta = 0.25, nIter = 100):
		self.eta = eta

		self.iter = nIter