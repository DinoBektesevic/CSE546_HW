"""
    TEST SCRIPT FOR POLYNOMIAL REGRESSION 1
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

from polyreg import learningCurve




#----------------------------------------------------
# Plotting tools

def plotLearningCurve(errorTrain, errorTest, regLambda, degree):
    """
        plot computed learning curve
    """
    minX = 3
    maxY = max(errorTest[minX+1:])

    xs = np.arange(len(errorTrain))
    plt.plot(xs, errorTrain, 'r-o')
    plt.plot(xs, errorTest, 'b-o')
    plt.plot(xs, np.ones(len(xs)), 'k--')
    plt.legend(['Training Error', 'Testing Error'], loc='best')
    plt.title('Learning Curve (d='+str(degree)+', lambda='+str(regLambda)+')')
    plt.xlabel('Training samples')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.ylim(top=maxY)
    plt.xlim((minX, 10))


def generateLearningCurve(X, y, degree, regLambda):
    """
        computing learning curve via leave one out CV
    """

    n = len(X)

    errorTrains = np.zeros((n, n-1))
    errorTests = np.zeros((n, n-1))

    loo = model_selection.LeaveOneOut()
    itrial = 0
    for train_index, test_index in loo.split(X):
        #print("TRAIN indices:", train_index, "TEST indices:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        (errTrain, errTest) = learningCurve(X_train, y_train, X_test, y_test, regLambda, degree)

        errorTrains[itrial, :] = errTrain
        errorTests[itrial, :] = errTest
        itrial = itrial + 1

    errorTrain = errorTrains.mean(axis=0)
    errorTest = errorTests.mean(axis=0)

    plotLearningCurve(errorTrain, errorTest, regLambda, degree)




#-----------------------------------------------

if __name__ == "__main__":
    '''
        Main function to test polynomial regression
    '''

    # load the data
    filePath = "data/polydata.dat"
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    X = allData[:, [0]]
    y = allData[:, [1]]

    # generate Learning curves for different params
    plt.figure(figsize=(15, 9), dpi=100)
    plt.subplot(2, 3, 1)
    generateLearningCurve(X, y, 1, 0)
    plt.subplot(2, 3, 2)
    generateLearningCurve(X, y, 4, 0)
    plt.subplot(2, 3, 3)
    generateLearningCurve(X, y, 8, 0)
    plt.subplot(2, 3, 4)
    generateLearningCurve(X, y, 8, .1)
    plt.subplot(2, 3, 5)
    generateLearningCurve(X, y, 8, 1)
    plt.subplot(2, 3, 6)
    generateLearningCurve(X, y, 8, 100)
    plt.show()
