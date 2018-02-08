import numpy as np
from scipy import linalg as la

# History:
# 2018-01-21 - JL - Create linreg class and the calcBeta method

# Consider implementing different methods to find the weights
# - normal equation
# - gradient descent
# - MLE with Newton-Rhapson method?

class linreg:

	def __init__(self, method = "normal"):
		self.weights = []

		if method == "normal":
			self.method = "normal"
		else:
			self.method = method

	def calcMeanReg(self, data):
		# get feature matrix attributes
		nData = np.shape(data)[0]
		nFeatVars = np.shape(data)[1]

		# add column for b0 to feature matrix if not there already
		if ~np.all(data[:, 0] == 1):
			data = np.concatenate((np.ones((nData, 1)), data), axis = 1)

		# calculated predicted y's
		y_hat = np.dot(data, self.weights)
		return y_hat

	def calcBeta(self, data, y):
		# get feature matrix attributes
		nData = np.shape(data)[0]
		nFeatVars = np.shape(data)[1]

		# transpose labels to fit feature matrix layout
		if np.shape(y)[0] == 1:
			y = y.T
		elif isinstance(y, np.ndarray):
			# labels = np.reshape(labels, (np.size(labels), 1))
			y = np.reshape(y, (nData, 1))

		# add column for b0 to feature matrix if not there already
		if ~np.all(data[:, 0] == 1):
			data = np.concatenate((np.ones((nData, 1)), data), axis = 1)

		if self.method == "normal":
			self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(data.T, data)), data.T), y)
		else:
			self.weights = np.empty((nFeatVars + 1, 1))

		y_hat = self.calcMeanReg(data)

		return y_hat