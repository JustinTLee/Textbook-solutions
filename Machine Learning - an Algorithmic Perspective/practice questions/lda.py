import numpy as np
from scipy import linalg as la

# The way Marsland describes LDA theory is not how he implmenents his actual
# code. Instead, the method implemented is from Pattern Recognition and Machine
# Learning by Christopher Bishop (p. 191-192).

# History:
# 2018-01-19 - JL - wrote meandev method, extractSW, and extractSb methods
# 2018-01-20 - JL - frustrated with how Marsland implemented the code differently
#                   from book's outline, so switched to Bishop's implementation,
#                   created LDA class

def meandev(data):
    # Create mean-deviation form of dataset
    data = data - np.mean(data, axis = 0)
    return data


def calculateCovMat(data, vectLabels):
    # Create the within-class and between-class covariance matrices

    # create mean deviation form of data:
    data = meandev(data)
    
    # Initialize empty matrices
    nFeatVar = np.shape(data)[1]
    matSW = np.zeros((nFeatVar, nFeatVar))
    matSB = np.zeros((nFeatVar, nFeatVar))

    # create a vector of unique classes to cycle through
    vectClass = np.unique(vectLabels)
    nClasses = np.size(vectClass)

    # create mean vector for entire dataset
    vectMean = np.reshape(np.mean(data, axis = 0), (nFeatVar, 1))

    for x in vectClass:
        # logically select all rows with class x
        boolClass = np.squeeze(vectLabels == x)

        # create class mean vector
        vectClassMean = np.reshape(np.mean(data[boolClass, :], axis = 0), (nFeatVar, 1))
        
        # create the covariance within-class matrix (SW)
        matSW += np.sum(boolClass)*np.cov(data[boolClass, :].T, bias = True)
        matSB += np.sum(boolClass)*np.dot((vectClassMean - vectMean), (vectClassMean - vectMean).T)

    matST = np.shape(data)[0]*np.cov(data.T, bias = True)

    return matSW, matSB

class lda:

    def __init__(self):
        self.weights = []

    def transformData(self, matFeat):
        matFeat = meandev(matFeat)
        return np.dot(matFeat, self.weights)

    def trainWeights(self, matFeat, vectLabels, reducedDim):
        matSW, matSB = calculateCovMat(matFeat, vectLabels)

        vectEigVal, matEigVect = la.eig(np.dot(la.inv(matSW), matSB))
        
        # sort eigenvectors by highest eigenvalue
        boolEigInd = np.argsort(vectEigVal)
        boolEigInd = boolEigInd[::-1]
        matEigVect = matEigVect[:, boolEigInd]
        self.eigvals = abs(np.real(vectEigVal[boolEigInd]))
        self.weights = matEigVect[:, :reducedDim]

        for k in range(np.size(self.eigvals)):
            eig_str = "Eigenvalue #" + str(k) + ": " + str(self.eigvals[k]) + " accounts for " + str(round(self.eigvals[k]/np.sum(self.eigvals)*100, 2)) + "% of data"
            print(eig_str)

        return self.transformData(matFeat)