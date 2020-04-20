'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:
    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        self.regLambda = reg_lambda
        self.degree = degree
        self.theta = None
        self._mean = None
        self._std = None


    def _expandToDegree(self, X, degree=None):
        """Expands the given column vector to a (n,d) matrix where elements are
        powers of values x_i including the zero-th order.

        [x1, ... x_n]^T = [[1, x1, x_1^2, ... x_1d^d],
                                       ...
                           [1, x_n, x_n^2, ... x_nd^d]]
        """
        if degree is None:
            degree = self.degree
        return (X[:,None]**np.arange(degree+1))[:, 0, :]


    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
        degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        self.degree = degree
        return self._expandToDegree(X, degree)[:, 1:]


    def standardize(self, X, mean=None, std=None):
        """Returns a standardized copy of the array using the given weights.

        Standardization is performed by offsetting by mean and dividing by
        variance on a per column basis.
        """
        mean = self._mean if mean is None else mean
        std = self._std if std is None else std
        standardized = []
        for row in X:
            standardized.append((row-mean)/std)
        return np.vstack(standardized)


    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        # expand to polynomial of degree d
        X_ = self.polyfeatures(X, self.degree)

        # standardize the matrix and remember the weights
        self._mean = np.mean(X_, axis=0)
        self._std = np.std(X_, axis=0)
        X_ = self.standardize(X_)

        # add a column of oens
        X_ = np.c_[np.ones([len(X), 1]), X_]

        # construct reg matrix
        reg_matrix = self.regLambda * np.eye(self.degree+1)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)
        return self.theta


    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        X_ = self.polyfeatures(X, self.degree)
        X_ = self.standardize(X_)
        X_ = np.c_[np.ones([len(X), 1]), X_]        
        return X_.dot(self.theta)


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape

    return errorTrain, errorTest
