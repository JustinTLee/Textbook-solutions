# LDA module

import numpy as np
from scipy import linalg as la
import csv

datafile = np.loadtxt(open('/home/jlee/Documents/datasets/iris/iris_proc.data'), delimiter = ",")
matData = datafile[:, 0:4]
vectLabels = datafile[:, 4]

vectClass = np.unique(vectLabels)
nClasses = np.size(vectClass)
nFeatVar = np.size(matData, axis = 1)
nData = np.size(matData, axis = 0)

matSW = np.zeros((nFeatVar, nFeatVar))
matSB = np.zeros((nFeatVar, nFeatVar))
C = np.cov(np.transpose(matData))

# create mean vector for entire dataset
vectMean = np.reshape(np.mean(matData, axis = 0), (nFeatVar, 1))

for x in vectClass:
    # create probabilities based on frequencies for each class
    pC = np.sum(vectLabels == x)/np.size(vectLabels)
    
    # logically select all rows with class x
    boolClass = np.squeeze(vectLabels == x)
        
    # create the variance within-class matrix (SW)
    matSWC = np.cov(np.transpose(matData[boolClass, :]))
    matSW = matSW + pC*np.cov(np.transpose(matData[boolClass, :]))
    
    # mean-deviation from data mean
    vectMeanDev = np.reshape(np.mean(matData[boolClass, :], axis = 0), (nFeatVar, 1)) - vectMean

    # create variance between-class matrix (SB)
    matSB = matSB + np.dot(vectMeanDev, np.transpose(vectMeanDev))

# solve generalized eigenvalue problem for the ratio of between-class to within-class (SB/SW)
vectEigVal, matEigVect = la.eig(la.inv(matSW).dot(matSB))

boolEigInd = np.argsort(vectEigVal)
boolEigInd = boolEigInd[::-1]
matEigVect = matEigVect[:, boolEigInd]
vectEigVal = vectEigVal[boolEigInd]