import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import time
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)

def load_mnist_dataset(path="data/mnist_data/"):
    """Loads MNIST data located at path.

    MNIST data are 28x28 pixel large images of numbers.

    Parameters
    ----------
    path : `str`
        path to the data directory

    Returns
    -------
    train : `torch.tensor`
        train data normalized to 1
    trainlabels : `torch.tensor`
        train data labels
    test : `torch.tensor`
        test data normalized to 1
    testLabels : `torch.tensor`
        test data labels
    """
    mndata = MNIST(path)

    train, trainLabels = map(torch.tensor, mndata.load_training())
    test, testLabels  = map(torch.tensor, mndata.load_testing())

    train = train/255.0
    test = test/255.0

    return (train.type(torch.DoubleTensor), trainLabels.long(),
        test.type(torch.DoubleTensor), testLabels.long())


def calculate_error(model, x, y):
    """Given a model, features and labels calculates the models missclassification
    error in percentage.

    Parameters
    ----------
    model: `func`
        Model, a function that takes in features and returns labels.
    x: `torch.tensor`
        Features
    y: `torch.tesnor`
        Labels

    Returns
    -------
    error: `float`
        Missclassification error.
    """
    yHat = model(x)
    predictions = torch.argmax(yHat, dim=1)
    return float(100*(predictions != y).float().mean())


def train(model, optimizer, lossFunc, x, y, batchSize, tolerance=1):
    """Given a model, optimizer, desirdata and batch size
    trains the model.

    Parameters
    ----------
    model: `func`
        Model, a function that takes in features and returns labels.
    optimizer: `class`
        Optimizer, one of torch.optim classes (e.g. torch.optim.Adam)
    lossFunc: `class`
        Loss function, one of torch.nn classes (e.g. torch.nn.CrossEntropyLoss)
    x: `torch.tensor`
        Features
    y: `torch.tensor`
        Labels
    batchSize: `int`
        Size of batches used in training.
    tolerance: `float`
        When error becomes smaller than tolerance, iterations are terminated.

    Returns
    -------
    errors: `list`
        Errors per epoch
    losses: `list`
        Loss per epoch

    Notes
    -----
    Model is trained for 1000 epoch. Error and loss for each epoch is recorded.
    """
    indices = np.arange(len(x))
    nIter, converged = 0, False
    errors, losses = [], []
    for i in range(1000):
        indexList = np.random.permutation(np.arange(len(x)))
        batches = np.split(indices, batchSize)
        for batch in batches:
            data = x[batch]
            labels = y[batch]

            fitted = model(data)

            optimizer.zero_grad()
            loss = lossFunc(fitted, labels)
            loss.backward()
            optimizer.step()

            newError = calculate_error(model, x, y)
            errors.append(newError)
            losses.append(float(loss))

            if newError < tolerance:
                converged = True
                break

        if converged:
            break

    return errors, losses


def define_model(w0, b0, w1, b1, w2=None, b2=None, sigma=nn.ReLU(), which="a"):
    """A closure that returns a model with the given layer weights and offsets.
    Defines two models "a" and "b" (aka shallow and deep network) for the two
    networks defined in the problem. Models take in a single input, the
    features.

    Parameters
    ----------
    w*: `torch.tensor`
       Layer weights, number indicates the layer depth. Acceptable depths are
       from 0 to 2 (e.g. w0, w1, w2).
    b*: `torch.tensor`
       Layer offsets, number indicates layer depth. Acceptable depths are
       from 0 to 2 (e.g. b0, b1, b2).
    sigma: `func`
        Activation function, e.g. `torch.nn.ReLu()`
    which: `str`, optional
        Which model to return, accepts ``a`` or ``b``. Default: ``a``

    Returns
    -------
    model: `func`
        Singgle argument function, the model.
    """
    def A5a_model(x):
        """Two layer model: ReLU(W0 X + b0) W2 + B2 """
        inner = x@w0 + b0
        return sigma(inner) @ w1 + b1

    def A5b_model(x):
        """Three layer model: ReLu( W1 (ReLU(W0 X + b0) W2 + B2) + B1) W2 +B2"""
        innest = x@w0 + b0
        inner = sigma(innest) @ w1 + b1
        return sigma(inner) @ w2 + b2

    if which == "a":
        return A5a_model
    elif which == "b":
        return A5b_model
    else:
        raise AttributeError("You missed!, How could you miss!? He was three feet "
                             "in front of you!\n\t - Mushu in a snowy pass, 1998")

def plot_paths(losses, errors):
    """Plots given losses and errors as a function of epoch.

    Paramters
    ---------
    losses: `list`
        Loss per epoch
    errors: `list`
        Error per epoch
    """
    fig, axes = plt.subplots(2,1, sharex=True)

    x = np.arange(len(losses))

    axes[0].plot(x, losses)
    axes[0].set_ylabel("Loss (cross entropy)")

    axes[1].plot(x, errors)
    axes[1].set_ylabel("Missclassification (%)")
    axes[1].set_xlabel("N. iteration")

    plt.show()


def A5a(learnRate=0.05, batchSize=6):
    """Defines weights and offsets of a 2 layered neural network, trains it,
    calculates train and test accuracy and losses and plots training losses and
    errors per epoch

    Parameters
    ----------
    learnRate: `float`, optional
        Learning rate, default 0.05
    batchSize: `int`, optional
        Training batch size, default 6
    """
    n, d, h, k = 28, 28**2, 64, 10

    tDim, lDim = 1/n, 1/np.sqrt(h)
    w0 = torch.DoubleTensor(d, h).uniform_(-tDim, tDim).requires_grad_()
    b0 = torch.DoubleTensor(1, h).uniform_(-tDim, tDim).requires_grad_()
    w1 = torch.DoubleTensor(h, k).uniform_(-lDim, lDim).requires_grad_()
    b1 = torch.DoubleTensor(1, k).uniform_(-lDim, lDim).requires_grad_()

    trainData, trainLabels, testData, testLabels = load_mnist_dataset()
   
    optimizer = torch.optim.Adam([w0, b0, w1, b1], lr=learnRate)
    model = define_model(w0, b0, w1, b1)
    loss = nn.CrossEntropyLoss()

    errors, losses = train(model, optimizer, loss, trainData, trainLabels,
                           batchSize)

    # calculate loss and accuracy on testing set
    yHat = model(testData)
    testLoss = loss(yHat, testLabels)

    print(f"Accuracy (train): {100 - errors[-1]}")
    print(f"Loss (train): {losses[-1]}")

    print(f"Accuracy (test) {100 - calculate_error(model, testData, testLabels)}")
    print(f"Loss (test): {float(testLoss)}")

    plot_paths(losses, errors)


def A5b(learnRate=0.05, batchSize=6):
    """Defines weights and offsets of a 3 layered neural network, trains it,
    calculates train and test accuracy and losses and plots training losses and
    errors per epoch

    Parameters
    ----------
    learnRate: `float`, optional
        Learning rate, default 0.05
    batchSize: `int`, optional
        Training batch size, default 6
    """

    n, d, h0, h1, k = 28, 28**2, 32, 32, 10

    tDim, l0Dim, l1Dim = 1/n, 1/np.sqrt(h0), 1/np.sqrt(h1)
    w0 = torch.DoubleTensor(d, h0).uniform_(-tDim, tDim).requires_grad_()
    b0 = torch.DoubleTensor(1, h0).uniform_(-tDim, tDim).requires_grad_()
    w1 = torch.DoubleTensor(h0, h1).uniform_(-l0Dim, l0Dim).requires_grad_()
    b1 = torch.DoubleTensor(1, h1).uniform_(-l0Dim, l0Dim).requires_grad_()
    w2 = torch.DoubleTensor(h1, k).uniform_(-l1Dim, l1Dim).requires_grad_()
    b2 = torch.DoubleTensor(1, k).uniform_(-l1Dim, l1Dim).requires_grad_()

    trainData, trainLabels, testData, testLabels = load_mnist_dataset()
   
    optimizer = torch.optim.Adam([w0, b0, w1, b1, w2, b2], lr=learnRate)
    model = define_model(w0, b0, w1, b1, w2, b2, which="b")
    loss = nn.CrossEntropyLoss()

    errors, losses = train(model, optimizer, loss, trainData, trainLabels,
                           batchSize)

    # calculate loss and accuracy on testing set
    yHat = model(testData)
    testLoss = loss(yHat, testLabels)

    print(f"Accuracy (train): {100 - errors[-1]}")
    print(f"Loss (train): {losses[-1]}")

    print(f"Accuracy (test) {100 - calculate_error(model, testData, testLabels)}")
    print(f"Loss (test): {float(testLoss)}")

    plot_paths(losses, errors)


if __name__ == "__main__":
    A5a()
    print()
    A5b()
