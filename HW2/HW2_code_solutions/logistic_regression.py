import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg

from mnist import MNIST


def separate(data, labels, *args):
    """Returns only data and labels that match the given values.

    Parameters
    ----------
    data: `np.array`
        Feature space data.
    labels: `np.array`
        Labels associated with the given features.
    args:
        Values of labels we want to separate data and labels on.

    Returns
    -------
    data: `np.array`
        The data for which labels match the given args.
    labels: `np.array`
        Labels that mathed the given args.
    """
    mask = [False]*len(labels)
    for arg in args:
        mask = np.logical_or(mask, labels == arg)
    return data[mask], labels[mask]


def encode_labels(labels, matchVals, encodeVals):
    """For given labels set values to elements of setVals where they match
    their respective matchVals.

    Parameters
    ----------
    matchVals: `tuple` or `list`
        Iterable of values labels must match in order to be set to their
        respective encoding values.
    encodeVals: `tuple` or `list`
        Encoding values assigned to matching values.
    """
    labels = labels.astype(int)
    for matchVal, encodeVal in zip(matchVals, encodeVals):
        labels[labels == matchVal] = encodeVal
    return labels


def mnist_numbers(path="data/mnist_data/", numbers=(2,7), encodeVals=(-1, 1)):
    """Loads MNIST data, located at path, normalizes, separates the desired
    numbers and encodes the matching data labels with given values.

    MNIST data are 28x28 pixel large images of letters from 0 to 9.

    Parameters
    ----------
    path: `str`
        path to the data directory
    numbers: `tuple` or `list`, optional
        Iterable of numbers that will be separated from the total dataset.
        Default: (2, 7)
    encodeVals: `tuple` or `list`, optional
        Encoding values assigned to each of the numbers separated.
        Default: (-1, 1)

    Returns
    -------
    train : `np.array`
        Train data normalized to 1
    trainlabels : `np.array`
        Encoded train data labels
    test : `np.array`
        Test data normalized to 1
    testLabels : `np.array`
        Encoded test data labels

    Notes
    -----
    The numbers and the encoding values must match in length and map according
    to their index. For example the default values encode number 2 labels as -1
    and number 7 labels as 1.
    Labels are encoded up to the shortest match to numbers, ignoring the rest.
    """
    mndata = MNIST(path)

    train, trainLabels = map(np.array, mndata.load_training())
    test, testLabels  = map(np.array, mndata.load_testing())

    train = train/255.0
    test = test/255.0

    train, trainLabels = separate(train, trainLabels, *numbers)
    trainLabels = encode_labels(trainLabels, numbers, encodeVals)
    test, testLabels = separate(test, testLabels, *numbers)
    testLabels = encode_labels(testLabels, numbers, encodeVals)

    return train, trainLabels, test, testLabels


def J(x, y, w, b, lambd):
    """Calculates the regularized negative log likelihood function:

        J(w, b) = 1/n \sum log( 1/mu_i + lambda ||w||^2

    see `mu` for `mu_i(w, b)` for more.

    Parameters
    ----------
    x: `np.array`
        Feature space data.
    y: `np.array`
        Labels associated with the given features.
    w: `np.array`
        Model weights.
    b: `float`
        Model offset
    lambd: `float`
        Regularization parameter.

    Returns
    -------
    J: `float`
        Objective, the regularized negative log likelihood.
    """
    n, d = x.shape
    exponential = np.exp(-(y*b + y*np.dot(x, w)))
    log = np.log10(1 + exponential) / (n*np.log(10))
    regularization = lambd * np.dot(w.T, w)
    return  np.sum(log) + regularization


def mu(x, y, w, b):
    """Calculates the value of the substitution expression:

        mu(w, b)  = 1 / ( 1+ exp(y(b + x^Tw)))

    that makes gradient calculations easier.

    Parameters
    ----------
    x: `np.array`
        Feature space data.
    y: `np.array`
        Labels associated with the given features.
    w: `np.array`
        Model weights.
    b: `float`
        Model offset

    Returns
    -------
    mu : `float`
        Value of the substitution expression
    """
    exponential = np.exp(-y*b - y*np.dot(x, w))
    return 1 / (1+exponential)


def grad_w_J(x, y, w, b, lambd):
    """Calculates gradient of the regularized negative log likelihood function
    with respect to the weights:

        J(w, b) = 1/n \sum y_i x_i (1-mu_i) + 2 lambda w

    see `mu` for more on `mu_i`

    Parameters
    ----------
    x: `np.array`
        Feature space data.
    y: `np.array`
        Labels associated with the given features.
    w: `np.array`
        Model weights.
    b: `float`
        Model offset
    lambd: `float`
        Regularization parameter.

    Returns
    -------
    grad_w_J: `float`
        Gradient of objective with respect to w
    """
    n, d = x.shape
    mus = mu(x, y, w, b) - 1
    # the trick to performing row-wise multiplication is to match the axis
    # sizes of the vectors and arrays by adding a dummy axis
    firstTerm = y[:, np.newaxis] * x * mus[:, np.newaxis]
    secondTerm = 2*lambd*w
    return np.sum(firstTerm, axis=0) / (n*np.log(10)) + secondTerm


def grad_b_J(x, y, w, b):
    """Calculates gradient of the regularized negative log likelihood function
    with respect to the offset:

        J(w, b) = 1/n \sum y_i (1-mu_i)

    see `mu` for more on `mu_i`

    Parameters
    ----------
    x: `np.array`
        Feature space data.
    y: `np.array`
        Labels associated with the given features.
    w: `np.array`
        Model weights.
    b: `float`
        Model offset

    Returns
    -------
    grad_b_J: `float`
        Gradient of objective with respect to b
    """
    n, d = x.shape
    mus = mu(x, y, w, b) - 1
    # row-wise multiplication
    firstTerm = y[:, np.newaxis] * x * mus[:, np.newaxis]
    return np.sum(firstTerm) / (n*np.log(10))


def classify(x, w, b):
    """Returns binary classification of the data.

    Parameters
    ----------
    x: `np.array`
        Feature space we want classified.
    w: `np.array`
        Model weights.
    b: `float`
        Model offset.

    Returns
    -------
    classes: `np.array`
        Array of binary positive or negative values (1 or -1) representing the
        classification of the model.
    """
    return np.sign(b + np.dot(x, w))


def count_missclassified(data, w, b, trueLabels):
    """Given features data, weights offsets and true labels counts the number
    of points missclassified by the model.

    Parameters
    ----------
    data: `np.array`
        Feature space to classify.
    w: `np.array`
        Model weights.
    b: `float`
        Model offset.
    trueLabels: `np.array`
        True labels for the features.

    Returns
    -------
    count: `int`
        Number of missclassified points.
    """
    return  np.sum(np.abs( classify(data, w, b) - trueLabels )) / (2*len(trueLabels))


def gradient_descent(data, labels, lambd, step, nIter=10, stochastic=False,  batchSize=1):
    """Performs gradient, or stochastic gradient, descent on the given data.

    Parameters
    ----------
    data: `np.array`
        Label space to learn the weights on.
    labels: `np.array`
        Labels for the given data.
    lambd: `float`
        Regularization parameter.
    step: `float`
        Steps size to take in the direction of the gradient.
    nIter: `int`, optional
        Number of iterations to preform, note that the function does not test
        for convergence so ensure sufficient number of steps. Default: 10
    stochastic: `bool`, optional
        If True preforms stochastic gradient descent. Default: False
    batchSize: `int`, optional
        Size of the data point set on which gradient estimate is performed on.
        Only used if stochastic is True. Default: 1

    Returns
    -------
    w: `np.array`
        Learned weights after nIter iterations.
    b: `float`
        Learned offsets after nIter iterations.
    calcJ: `np.array`
        Value of ojective in each step.
    missclassified: `np.array`
        Number of missclassified features in each step.
    wSteps: `np.array`
        Leaned weights in each step, used to estimate J and missclass. on test.
    bSteps: `np.array`
        Learned offsets in each step, used to estimate J and missclass. on test.
    """
    n, d = data.shape
    w = np.zeros(d)
    b = 0

    calcJ, missclassified, wSteps, bSteps = [], [], [], []
    for i in range(nIter):
        # append results befoe we alter values to capture zeroth element correctly
        wSteps.append(w)
        bSteps.append(b)
        calcJ.append(J(data, labels, w, b, lambd))
        missclassified.append(count_missclassified(data, w, b, labels))

        # stochastic picks some n random elements for the gradient calculation.
        # otherwise perform regular gradient estimate over all points
        if stochastic:
            idxs = np.random.permutation(np.arange(n))[:batchSize]
            w = w - step * grad_w_J(data[idxs], labels[idxs], w, b, lambd)
            b = b - step * grad_b_J(data[idxs], labels[idxs], w, b)
        else:
            w = w - step * grad_w_J(data, labels, w, b, lambd)
            b = b - step * grad_b_J(data, labels, w, b)

    return w, b, calcJ, missclassified, wSteps, bSteps


def plot(ax, x, y, label="", xlabel="", ylabel="", title="", xlog=False, lc='black', lw=1):
    """Plots a line on given axis.

    Parameters
    ----------
    ax: `matplotlib.pyplot.Axes`
       Axis to plot on.
    x: `np.array`
       X axis values
    y: `np.array`
       Y axis values
    label: `str`, optional
       Line label
    xlabel: `str`, optional
       X axis label
    ylabel: `str`, optional
       Y axis label
    title: `str`, optional
       Axis title
    xlog : `bool`, optional
       X axis scaling will be logarithmic
    lc : `str`, optional
       Line color
    lw: `int` or `float`, optional
       Line width

    Returns
    ------
    ax: `matplotlib.pyplot.Axes`
        Modified axis.
    """
    ax.set_title(title)
    ax.plot(x, y, label=label, color=lc, linewidth=lw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlog:
        ax.set_xscale('log')
    return ax


def A6abc(nIter, lambd, step, stochastic=False, batchSize=1, title="Gradient Descent"):
    """Reads MNIST dataset separates numbers 2 and 7, encodes -1 and 1 labels
    for the numbers respectively, performs gradient descent with the given
    parameters, estimates objective and missclassified labels in each step for
    both train and test datasets.

    Plots the values of objective and missclassifications for both train and
    test datasets.

    Parameters
    ----------
    nIter: `int`, optional
        Number of iterations to preform, note that the function does not test
        for convergence so ensure sufficient number of steps. Default: 10
    lambd: `float`
        Regularization parameter.
    step: `float`
        Steps size to take in the direction of the gradient.
    stochastic: `bool`, optional
        If True preforms stochastic gradient descent. Default: False
    batchSize: `int`, optional
        Size of the data point set on which gradient estimate is performed on.
        Only used if stochastic is True. Default: 1
    title: `str`
        Title of the produced plots.
    """
    trainData, trainLabels, testData, testLabels = mnist_numbers()

    # annoyingly we have to re-iterate or live with ugly grad_desc func.
    w, b, trainJ, testMisslbls, wSteps, bSteps =  gradient_descent(trainData, trainLabels, lambd,
                                                                   step, nIter=nIter, stochastic=stochastic,
                                                                   batchSize=batchSize)

    testJ, trainMisslbls = [], []
    for wi, bi in zip(wSteps, bSteps):
        testJ.append(J(testData, testLabels, wi, bi, lambd))
        trainMisslbls.append(count_missclassified(testData, wi, bi, testLabels))

    print(w)
    print(f"{title}")
    print(f"    Converged to offset b={b:.4f}")
    print(f"    Objective converged for train to J={trainJ[-1]:.4f} and test J={testJ[-1]:.4f}")

    iters = np.arange(nIter)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1 = plot(ax1, iters, trainJ, label="Training objective f.", lc="orange")
    ax1 = plot(ax1, iters, testJ, label="Test objective f.", title=title,
               xlabel="Iteration number", ylabel="J(w,b)", lc="blue")
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2 = plot(ax2, iters, testMisslbls, label="Training missclassifications.",
               lc="gray")
    ax2 = plot(ax2, iters, trainMisslbls, label="Test missclassifications.",
               title="title", xlabel="Iteration number",
               ylabel="N. missclassified labels.")
    ax2.legend()
    plt.show()


def A6b(nIter=200, lambd=0.1, step=0.01):
    """Calls A6abc with parameters specified in problem A6 b"""
    A6abc(nIter, lambd, step)


def A6c(nIter=200, lambd=0.1, step=0.01, stochastic=True, batchSize=1):
    """Calls A6abc with parameters specified in problem A6 c"""
    A6abc(nIter, lambd, step, stochastic, batchSize, title="Stochastic Gradient Descent.")


def A6d(nIter=200, lambd=0.1, step=0.01, stochastic=True, batchSize=100):
    """Calls A6abc with parameters specified in problem A6 d"""
    A6abc(nIter, lambd, step, stochastic, batchSize, title="Stochastic Gradient Descent.")


def A6():
    A6b()
    A6c()
    A6d()
    pass


if __name__ == "__main__":
    A6()
