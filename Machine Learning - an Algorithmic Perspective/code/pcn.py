import numpy as np
import datetime as dt

# History:
# 2018-01-20 - JL - started pcn class, created instantiation
#				  - created thresholding function (gating)
#				  - created label prediction function
#                 - created training function

class pcn:

	def __init__(self, nNeurons, seed = None, iter = None, thresh_type = 'linear'):
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
		if thresh_type != 'linear':
			self.thresh_type = thresh_type
		else:
			self.thresh_type = 'linear'
		
		# set seed for class
		np.random.seed(self.seed)

		# generate eta as random number between 0.1 to 0.4, as stated in p. 46
		self.eta = np.random.uniform(0.1, 0.4, 1)

	def thresholdH(self, hij):
		# use various activation functions to threshold result

		# simple binary threshold to see if hij is above or below 0
		if self.thresh_type == 'linear':
			yi = np.where(hij > 0, 1, 0)

		# logistic threshold with boundary at 0.5
		elif self.thresh_type == 'logistic':
			logit_ij = 1/(1 + np.exp(-hij))
			yi = np.where(logit_ij > 0.5, 1, 0)

		# if any-non empty string other than the possible options, return zero array
		else:
			yi = np.zeros((np.shape(hij)[0], 1))

		return yi

	def predictLabels(self, data):
		# create the bias column if it doesn't exist
		if np.all(data[:, 0] != -1):
			bias = -1*np.ones((nData, 1)) # bias weight
			data = np.concatenate((bias, data), axis = 1)

		# multiply input by weights to get hij
		hij = np.dot(data, self.weights)

		# threshold hij according to some function
		yi = self.thresholdH(hij)

		return yi

	def trainWeights(self, data, labels):
		# get input data attributes
		nData = np.shape(data)[0]
		nFeatVar = np.shape(data)[1]

		# transpose labels to fit feature matrix layout
		if np.shape(labels)[0] == 1:
			labels = labels.T
		elif isinstance(labels, np.ndarray):
			# labels = np.reshape(labels, (np.size(labels), 1))
			labels = np.reshape(labels, (nData, 1))

		# add bias node
		bias = -1*np.ones((nData, 1)) # bias weight
		data = np.concatenate((bias, data), axis = 1)

		# create weights
		self.weights = np.random.uniform(-1, 1, (nFeatVar + 1, self.nNeurons))

		for k in range(self.iter):
			# create label predictions
			yi = self.predictLabels(data)

			# update the weights
			self.weights -= np.squeeze(self.eta)*data.T*(yi - labels)

		yi = self.predictLabels(data)

		return yi