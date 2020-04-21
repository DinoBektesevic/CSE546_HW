import numpy as np
from mnist import MNIST
from scipy import linalg


def load_mnist_dataset(path="data/mnist_data/"):
    """Loads MNIST data located at path.

    MNIST data are 28x28 pixel large images of letters.

    Parameters
    ----------
    path : `str`
        path to the data directory

    Returns
    -------
    train : `np.array`
        train data normalized to 1
    trainlabels : `np.array`
        train data labels
    test : `np.array`
        test data normalized to 1
    testLabels : `np.array`
        test data labels    
    """
    mndata = MNIST("data/mnist_data/")

    train, trainLabels = map(np.array, mndata.load_training())
    test, testLabels  = map(np.array, mndata.load_testing())

    train = train/255.0
    test = test/255.0

    return train, trainLabels, test, testLabels


def one_hot(length, index):
    """Given an index and length k returns an array where all elements are zero
    except the one at index location, where the value is 1.

    Parameters
    ----------
    length : `int`
        Length of the almost-zero array.
    index : `int`
        Index at which element value is set to 1

    Returns
    -------
    arr : `np.array`
        Array of zeros except for arr[index]=1. 
    """
    arr = np.zeros(length)
    arr[index] = 1
    return arr


def train(X, Y, lamb):
    """Given data, labels and regularization constant lambda solves 

    $$ W = (X^T X) + \lambda I $$

    to retrieve weights of our model.

    Parameters
    ----------
    X : `np.array`
        Data to fit to
    Y : `np.array`
        Data labes, a length 10 array where index of element with value 1 marks
        the number the number respective data point x represents.
    lamb : `float`
        Regularization parameter lambda.

    Returns
    -------
    wHat : `np.array`
        Matrix of weights that minimize the linear least squares.
    """
    n, d = X.shape

    a = np.dot(X.T, X) + lamb*np.eye(d)
    b = np.dot(X.T, Y)
    wHat = linalg.solve(a, b)

    return wHat


def predict(W, data, labelDim):
    """Given weights, data and the dimension of the labels space predicts what
    label is the data most likely representing.

    Parameters
    ----------
    W : `np.array`
        Array of weights of our model.
    data : `np.array`
        Array of data to classify
    labelDim : `int`
        Label space dimension

    Returns
    -------
    classifications : `np.array`
        Array of final predicted classifications of the data.
    """
    predictions = np.dot(data, W)
    # pick out only the most probably values, i.e. the maxima
    maxPredictions = np.argmax(predictions, axis=1)
    classifications = np.array([one_hot(labelDim, y) for y in maxPredictions])
    return classifications


def calc_success_fraction(W, data, labels):
    """Given weights, data and labels predicts the labels of the data and by
    comparing them to the given labels calculates the fraction of the predicted
    classifications that were correct and wrong as a

    fracWrong = (\sum |predicted - actualLabel|) / (2*N_data)
    fracCorrect = 1 - fracWrong

    Parameters
    ----------
    W : `np.array`
        Weights of our model
    data : `np.array`
        data we want to predict labels for
    labels : `np.array`
        labels of actual class the data

    Returns
    -------
    fracCorrect : `float`
        Fraction of correctly predicted labels
    fracWrong : `float`
        Fraction of incorrectly predicted labels 
    """
    n, d = data.shape
    labelDim = labels.shape[-1]

    wrong = np.sum(np.abs(predict(W, data, labelDim) - labels))
    # 2 is required because abs value will contribute double to the sum
    fracWrong = wrong/(2.0*n)
    fracCorrect = 1 - fracWrong

    return fracCorrect, fracWrong


def main(lambd=1e-4):
    """Given the dimension of label space and regularization parameter value
    trains a model on the MNIST train dataset, predicts the labels on the MNIST
    test dataset and calculates the fraction of wrongly predicted labels.

    Parameters
    ----------
    lambd : `float`
        Regularization parameter (lamda)

    Returns
    -------
    trainErr : `float`
        Training error, fraction of incorrectly labeled train data
    testErr : `float`
        Test error, fraction of incorrectly labeled test data
    """
    xTrain, trainLabels, xTest, testLabels = load_mnist_dataset()
    n, d = xTrain.shape
    labelDim = trainLabels.max() + 1 

    yTrain = np.array([one_hot(labelDim, y) for y in trainLabels])
    yTest  = np.array([one_hot(labelDim, y) for y in testLabels])

    wHat = train(xTrain, yTrain, lambd)

    trainErr = calc_success_fraction(wHat, xTrain, yTrain)[-1]
    testErr  = calc_success_fraction(wHat, xTest,  yTest)[-1]
    print(f"Train error: {trainErr}")
    print(f"Test error: {testErr}")

    return trainErr, testErr


if __name__ == "__main__":
    main()
