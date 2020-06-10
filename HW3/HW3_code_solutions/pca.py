import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import time


np.random.seed(0)


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

    train, trainLabels = map(np.array, mndata.load_training())
    test, testLabels  = map(np.array, mndata.load_testing())

    train = train/255.0
    test = test/255.0

    return train, trainLabels, test, testLabels


def calculate_errors(x, eigenvectors=None, transMatrix=None):
    """Calculates mean square error of PCA prediction.

    Paramters
    ---------
    x: `np.array`
        Features
    Vk: `np.array`, optional
        Eigenvectors
    transMatrix: `np.array`, None
        Transformation matrix, the dot product of eigenvectors with themselves
        trainsposed.

    Returns
    -------
    error: `float`
        Means square error of reconstruction.
    """
    if transMatrix is None:
        if eigenvectors is None:
            raise AttributeError("Need to supply Vk!")
        transMatrix = np.dot(eigenvectors, eigenvectors.T)
    return np.mean((x - np.dot(x, transMatrix))**2)


def plot_eigen_fraction(k, frac):
    """Plots eingenvalue fraction as a function of the total number of
    eingenvectors found during decomposition.

    Parameters
    ----------
    k: `int`
        Number of fitted eigenvectors
    frac: `float`
        Fraction of eigenvalues over total eigenvalue sum.
    """
    fig, ax = plt.subplots()
    ax.plot(k, frac)
    ax.set_xlabel("k")
    ax.set_ylabel("Eingenvalue fraction")
    ax.set_title("Eigenvalue fraction vs k")
    plt.show()


def plot_errors(k, train, test):
    """Plots test and train error as a function of the number of eigenvectors
    used in reconstruction.

    Parameters
    ----------
    k: `int`
        Number of used eigenvectors
    test: `np.array`
        Test error
    train: `np.array`
        Train error
    """
    fig, ax = plt.subplots()
    ax.plot(k, test, label="Test error")
    ax.plot(k, train, label="Train error")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean squared reconstruction error")
    ax.set_title("MSE vs k.")
    ax.legend()
    plt.show()


def plot_n_eigenvectors(n, eigenvectors, nXaxes=2, nYaxes=5):
    """Plots first n eigenvectors

    Parameters
    ---------
    n: `int`
        Number of vectors to plot
    eigenvectors: `np.array`
        Eigenvectors
    nXaxes: `int`, optional
        Number of figure axes in the x direction
    nYaxes: `int`, optional
        Number of figure axes in the y direction
    """
    fig, axes = plt.subplots(nXaxes, nYaxes)

    for ax, k, eigVec in zip(axes.ravel(), range(n), eigenvectors.T):
        ax.imshow(eigVec.reshape((28, 28)))
        ax.set_title(f"k={k}")
        ax.axis("off")

    plt.show()


def plot_pca(x, y, eigenvectors, mu, digits =(2, 6, 7), ks=(5, 15, 40, 100)):
    """Plots the original digits and their reconstruction for different number
    of used eigenvectors.

    Parameters
    ----------
    x: `np.array`
        Features
    y: `np.array`
        Labels
    eigenvectors: `np.array`
        Eigenvectors
    mu: `np.array`
        Fitted mu.
    eigenvectors: `np.array`
        Eigenvectors
    digits: `tuple`
        Digits to plot
    ks: `tuple`
        Tuple of integers declaring how many eigenvectors should be used in
        reconstruction.
    """
    fig, axes = plt.subplots(len(digits), len(ks)+1)
    if len(digits) == 1:
        axes = np.array([axes])

    idxDigits = [np.where(y==digit)[0][0] for digit in digits]

    for yax, digit, idxDigit in zip(axes[:, 0], digits, idxDigits):
        yax.imshow(x[idxDigit].reshape((28, 28)))
        yax.set_title(f"Original image (digit {digit})")
        yax.axis("off")

    for yax, digit, idxDigit in zip(axes[:, 1:], digits, idxDigits):
        for xax, k in zip(yax, ks):
            Vk = eigenvectors[:, :k]
            reconstruction = np.dot(Vk, np.dot(Vk.T, (x-mu.T)[idxDigit])).reshape((784, 1))
            reconstruction += mu
            xax.imshow(reconstruction.reshape((28, 28)))
            xax.set_title(f"Reconstructed (k= {k})")
            xax.axis("off")

    plt.show()


def pca():
    """Preforms PCA on MNIST dataset.

    Calculates prints some eigenvalues, prints the sum of all eigenvalues.
    Plots first 25 eigenvectors.
    Plots eigenvalue fraction.
    Calculates the test and train errors for reconstructions up to first 100
    eigenvectors. Plots them.
    Reconstructs certain digits for varying number of used eigenvectors. Plots
    them.
    """
    train, trainLabels, test, testLabels = load_mnist_dataset()

    n, d = train.shape
    I = np.ones((n, 1))

    mu = np.dot(train.T, I)/n
    sigElem = train - np.dot(I, mu.T)
    sigma = np.dot(sigElem.T, sigElem)/n

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    eigenvalues = eigenvalues[np.argsort(-1 * eigenvalues)]
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]

    totEigenSum = np.sum(eigenvalues)

    trainErrors, testErrors, eigenRatios = [], [], []
    eigenSum, k = 0, np.arange(100)
    for i in k:
        Vk = eigenvectors[:, :(i+1)]
        transMatrix = np.dot(Vk, Vk.T)
        trainErrors.append(calculate_errors(train, transMatrix=transMatrix))
        testErrors.append(calculate_errors(test, transMatrix=transMatrix))
        eigenSum += eigenvalues[i]
        eigenRatios.append(1 - (eigenSum/totEigenSum))

    for i in (1, 2, 10, 30, 50):
        print(f"{i}th eigenvalue: {eigenvalues[i-1]}")
    print(f"Sum of eigenvalues: {totEigenSum}")

    plot_n_eigenvectors(16, eigenvectors, nXaxes=4, nYaxes=4)
    plot_eigen_fraction(k, eigenRatios)
    plot_errors(k, trainErrors, testErrors)
    plot_pca(train, trainLabels, eigenvectors, mu)


if __name__ == "__main__":
    pca()
