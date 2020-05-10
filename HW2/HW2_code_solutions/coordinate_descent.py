import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def descent(x, y, lambd, tolerance=0.001, initW=None):
    """Preforms coordinate descent Lasso algorithm on the given data.

    Parameters
    ----------
    x : `np.array`
        The feature space. A matrix with n rows of feature vectors, each with d
        features.
    y : `np.array`
        A column vector of n model values.
    lambd: `float`
        Regularization parameter.
    tolerance: `float`
        Convergence is achieved when the absolute value of minimal difference
        of old weights and the new weights estimates is less than tolerance.
    initW: `np.array`
        Initial weights, a vector of d feature weights.

    Returns
    -------
    w: `np.array`
        New feature weights estimates. 
    """
    n, d = x.shape

    if initW is None:
        w = np.zeros(d)
    else:
        w = initW

    # precalculate values used in the loop in advance
    squaredX =  2.0 * x**2

    # ensure convergence is not met on first loop
    convergeCriterion = tolerance + 1
    while convergeCriterion > tolerance:
        # not optimal test, but fast
        oldMax = w.max()
        deltas = []

        # Algorithm 1 implementation
        b = np.mean(y - np.dot(x, w))
        for k in range(d):
            xk = x[:, k]
            ak = squaredX[:,k].sum()

            # ck sum must ignore k-th dimension so we set it to zero and use
            # dot product. This matches the definition of w too, so we can
            # leave it wth zero value unless smaller than -lambda or bigger
            # than lambda anyhow. 
            w[k] = 0
            delta = 0
            ck = 2.0 * np.dot(xk, y - (b + np.dot(x, w)))

            if ck < -lambd:
                delta = (ck + lambd) / ak
                w[k] = delta
            elif ck > lambd:
                delta = (ck - lambd) / ak
                w[k] = delta

            deltas.append(delta)
        # Find maximum difference between iterations
        convergeCriterion = abs(oldMax-max(deltas))
    return w


def generate_data(n, d, k, sigma):
    """Generates i.i.d. samples of the model:
        y_i = w^T x_i + eps
    where
        w_j = j/k if j in {1,...,k}
        w_j = 0 otherwise
    and epsilon is random Gussian noise with the given sigma and X are also
    drawn from a Normal distribution with sigma 1. 

    Parameters
    ----------
    n : `int`
        Number of samples drawn at random from the model.
    d : `int`
        Dimensionality of the feature space. 
    k : `int`
        Cutoff point after which elements of w are zero.

    Returns
    -------
    x : `np.array`
        n-by-d sized array of data.
    y : `np.array`
        Vector of n model values. 
    """
    # gaussian noise and data
    eps = np.random.normal(0, sigma**2, size=n)
    x = np.random.normal(size=(n, d))

    # weights
    w = np.arange(1, d + 1) / k
    w[k:] = 0

    # labels
    y = np.dot(x, w) + eps
    return x, y


def plot(ax, x, y, label="", xlabel="", ylabel="", title="", xlog=True, lc='black', lw=2):
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


def lambda_max(x, y):
    """The smallest value of regularization parameter lambda for which the
    w is entirely zero.

    Parameters
    ----------
    x : `np.array`
        The feature space. A matrix with n rows of feature vectors, each with d
        features.
    y : `np.array`
        A column vector of n model values.

    Returns
    -------
    lambda: `float`
        Smallest lambda for which w is entirely zero.
    """
    return np.max(2 * np.abs(np.dot((y - np.mean(y)), x)))


def A4_setup(n=500, d=1000, k=100, sigma=1):
    """Creates data as instructed by A4 problem, calculates the smallest value
    of regularization parameter for which w is zero and returns data parameters,
    data and calculated lambda.

    Parameters
    ----------
    n : `int`, optional
        Number of samples drawn at random from the model. Default: 500
    d : `int`, optional
        Dimensionality of the feature space. Default: 1000
    k : `int`, optional
        Cutoff point after which elements of w are zero. Default: 100
    sigma: `float`, optional
        STD of the noise Gaussian distribution that is added to the data (see
        generate_data). Default: 1.

    Returns
    -------
    x : `np.array`
        The feature space. A matrix with n rows of feature vectors, each with d
        features.
    y : `np.array`
        A column vector of n model values.
    maxLambda: `float`
        Lambda for which w is zero everywhere.
    params: `dict`
        Dictionary of parameters used to create the data (n, d, k and sigma).
    """
    params = {'n': n, 'd': d, "k": k, "sigma": sigma}
    x, y = generate_data(n, d, k, sigma)
    maxLambda = lambda_max(x, y)
    return x, y, maxLambda, params


def A4(nIter=20, tolerance=0.001):
    """Sets the data up as instructed by problem A4 and runs coordinate
    descent Lasso algorithm nIter times, each time decreasing regularization
    parameter lambda by a factor of 1.5.
    Plots the number of non-zero-features against used lambda and false vs
    true discovery rates.

    Displays plots.

    Parameters
    ----------
    nIter: `int`, optional
        Number of different regularization parameter iterations to run. Default
        is 20.
    tolerance: `float`, optional
        Coordinate descent tolerance, sets convergence criteria (see descent).
        Default: 0.001.
    """
    x, y, lambd, params = A4_setup()
    k = params['k']

    lambdas, numNonZeros, fdrs, tprs = [], [], [], []
    for i in range(nIter):
        w = descent(x, y, lambd, tolerance)

        nonZeros = np.count_nonzero(w)
        correctNonZeros = np.count_nonzero(w[:k])
        incorrectNonZeros = np.count_nonzero(w[k+1:])

        try:
            fdrs.append(incorrectNonZeros/nonZeros)
        except ZeroDivisionError:
            fdrs.append(0)
        tprs.append(correctNonZeros/k)

        lambdas.append(lambd)
        numNonZeros.append(nonZeros)

        lambd /= 1.5

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    plot(axes[0], lambdas, numNonZeros, xlabel="Lambda",
         ylabel="Number of non zero features",
         title="N of non-zero features VS lambda.")
    plot(axes[1], fdrs, tprs, xlabel="False Discovery rate",
         ylabel="True Discovery rate", title="True VS False Discovery rates.",
         xlog=False)
    plt.show()


def A5_setup(trainFile="crime-train.txt", testFile="crime-test.txt"):
    """Reads the crime train and test data into a pandas dataframe and
    calculates maximum lambda value.

    Parameters
    ----------
    trainFile: `str`
        Path to file containing training data.
    testFile: `str`
        Path to file containing test data.

    Returns
    -------
    xTrain: `np.array`
        Training feature set.
    yTrain: `np.array`
        Label set, the response variable, the thing we model.
    lMaxTrain: `float`
        Maximal value of lambda for which all weights are zero.
    xTest: `np.array`
        Testing feature set.
    yTest: `np.array`
        Label set, the response variable, the thing we model.
    lMaxTest: `float`
        Maximal value of lambda for which all weights are zero.
    """
    train = pd.read_table("crime-train.txt")
    test = pd.read_table("crime-test.txt")

    yTrain = train["ViolentCrimesPerPop"]
    xTrain = train.drop("ViolentCrimesPerPop", axis=1)
    lMaxTrain = lambda_max(xTrain, yTrain)

    yTest = test["ViolentCrimesPerPop"]
    xTest = test.drop("ViolentCrimesPerPop", axis=1)
    lMaxTest = lambda_max(xTest, yTest)

    return xTrain, yTrain, lMaxTrain, xTest, yTest, lMaxTest


def mean_square_error(x, y, w):
    """For given data x, response y and weights w calculates the mean square
    error of the model (the square of difference of predicted VS actual).

    Parameters
    ----------
    x: `np.array`
        Feature set.
    y: `np.array`
        Label set
    w: `np.array`
        Weights.

    Returns
    -------
    mse: `float`
        Mean square error.
    """
    a = y - np.dot(x, w)
    return a.T @ a


def A5ab():
    """Sets the data up as instructed by problem A5 and runs coordinate
    descent Lasso algorithm untill the change in regularization parameter is
    smaller than 0.01. Each iteration decreases regularization parameter by a
    factor of 2.

    Plots the number of non-zero-features and mean square error against lambda.
    Displays plots.

    Parameters
    ----------
    nIter: `int`, optional
        Number of different regularization parameter iterations to run. Default
        is 20.
    tolerance: `float`, optional
        Coordinate descent tolerance, sets convergence criteria (see descent).
        Default: 0.001.
    """
    xTrain, yTrain, lMaxTrain, xTest, yTest, lMaxTest = A5_setup()

    # initialize weights, rename lambda so we don't alter original
    n, d = xTrain.shape
    w = np.zeros(d)
    lambd = lMaxTrain

    lambdas, numNonZeros, sqrErrTrain, sqrErrTest = [], [], [], []
    while lambd > 0.01:
        w = descent(xTrain.values, yTrain.values, lambd, initW=w)
        numNonZeros.append(np.count_nonzero(w))
        lambdas.append(lambd)
        sqrErrTrain.append(mean_square_error(xTrain.values, yTrain.values, w))
        sqrErrTest.append(mean_square_error(xTest.values, yTest.values, w))
        lambd /= 2.0

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot(ax1, lambdas, numNonZeros, xlabel="Lambda",
         ylabel="Number of non zero features",
         title="N of non-zero features VS lambda.")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot(ax2, lambdas, sqrErrTrain, label="MSE Train", lc="gray")
    plot(ax2, lambdas, sqrErrTest, label="MSE Test", xlabel="Lambda",
         ylabel="Mean Square Error", title="Mean Square Error VS lambda.")
    ax2.legend()
    plt.show()


def A5c():
    """Sets the data up as instructed by problem A5 and runs coordinate
    descent Lasso algorithm with regularization value of 30. 

    Prints the weights and the names of the features with most positive, most
    negative value and a sorted table of features with non-zero values.

    Parameters
    ----------
    nIter: `int`, optional
        Number of different regularization parameter iterations to run. Default
        is 20.
    tolerance: `float`, optional
        Coordinate descent tolerance, sets convergence criteria (see descent).
        Default: 0.001.
    """
    xTrain, yTrain, lMaxTrain, xTest, yTest, lMaxTest = A5_setup()

    # initialize weights, rename lambda so we don't alter original
    n, d = xTrain.shape
    w = np.zeros(d)
    lambd = 30

    lambdas, numNonZeros, sqrErr = [], [], []
    w = descent(xTrain.values, yTrain.values, lambd, initW=w)
    numNonZeros.append(np.count_nonzero(w))
    lambdas.append(lambd)

    idxMaxW = w == max(w)
    idxMinW = w == min(w)
    print(f"Maximal value of w {w[idxMaxW][0]:.4f} belongs to feature {xTrain.columns[idxMaxW][0]}")
    print(f"Maximal value of w {w[idxMinW][0]:.4f} belongs to feature {xTrain.columns[idxMinW][0]}")
    print()
    print("Feature Name              |  Magnitude")
    print("-----------------------------------------")
    idxSorted = np.argsort(w)
    idxNonZeroSorted = np.nonzero(w[idxSorted])
    for feature, w_i in zip(xTrain.columns[idxSorted][idxNonZeroSorted], w[idxSorted][idxNonZeroSorted]):
        print(f"{feature:25} | {w_i:4.8f}")


def A5():
    """Runs A5ab and A5c functions."""
    A5ab()
    A5c()


if __name__ == "__main__":
    A4()
    A5()
