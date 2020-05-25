import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from mnist import MNIST

np.random.seed(0)

class Kernel:
    """Base class used to define a generic, actionless, kernel (K0) to use in
    ridge regression defined as:

        a = argmin ||K*a - y||^2 + l*a^T*K*a
    where a is alpha, l - lambda and K the kernel itself and K=k(x_i, x_j) an
    kernel action. Kernel action is defined by its eval method, which must be
    overriden by subclass. 

    Attributes
    ----------
    x: `np.array`, optional
        Feature space.
    y: `np.array`, optional
        Labels. 
    lambd: `float`, optional
        Regularization parameter
    hyperparams: `dict`, optional
        A dictionary of hyperparameters.
    alpha: `dict`
        Lerned predictor (learned weights).

    Notes
    -----
    Fitting/training the kernel will update, replace, its features and labels
    with the ones given for training.

    Hyperparameters must be registered in the hyperparameters dictionary when
    inheriting. 
    """
    def __init__(self, x=None, y=None, lambd=None, **kwargs):
        self.update(x=x, y=y, lambd=lambd, **kwargs)
        self.alpha = None

    def update(self, **kwargs):
        """Update given class attributes (x, y, lambd, hyperparams)."""
        x = kwargs.pop("x", None)
        y = kwargs.pop("y", None)
        lambd = kwargs.pop("lambd", None)
        hyperparams = kwargs

        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if lambd is not None:
            self.lambd = lambd
        if len(hyperparams) != 0:
            self.hyperparams = hyperparams

    def eval(self, *args, **kwargs):
        """Defines kernel action, i.e. given an x_i, x_j evaluates the kernel.
        Needs to be an vectorized operation, such that supplying two vectors
        with dimensions n and m returns n-by-m matrix.
        Supplying a single vector x as both arguments should returns a diagonal
        matrix.
        """
        raise NotImplementedError

    def fit(self, x, y):
        """Given features x and labels y, updates the class attributes and
        preforms ridge regression. Stores the learned predictor in the alpha
        attribute.

        Parameters
        ----------
        x: `np.array`
            Features to train on.
        y: `np.array`
            Labels for corresponding features.
        """
        self.update(x=x, y=y)
        K = self.eval(x, x)
        self.alpha = np.linalg.solve(K + self.lambd * np.eye(len(K)), y)

    def predict(self, x):
        """Using learned weights and given features predicts the labels.

        Parameters
        ----------
        x: `np.array`
            Features.

        Returns
        --------
        y: `np.array`
            Predicted labels

        Raises
        ------
        AttributeError
            When kernel has not been trained and thus has no alpha attributed.
            Call fit with the features and associated labels to train the
            kernel.
        """
        if self.alpha is None:
            raise AttributeError("Kernel has not been trained.")
        basis = self.eval(self.x, x)
        return np.dot(self.alpha, basis)

    def score(self, x, y):
        """Predicts the labels of the given features and compares them to the
        given truth (true labels). Associates a score to the \"goodness\" of
        prediction.

        Parameters
        ----------
        x: `np.array`
            Features for which labels will be predicted.
        y: `np.array`
            Truth, true labels for the corresponding features.

        Returns
        -------
        score: `float`
            Mean of the square of differences in predicitons, the score of
            the goodness of predictions.
        """
        return np.mean((self.predict(x) - y)**2)

    def plot_basis(self, ax, x):
        basis = self.eval(self.x, x)
        for a, b in zip(self.alpha, basis):
            ax.plot(x, a*b)
        return ax


class PolynomialKernel(Kernel):
    """Class implementing the polynomial kernel."""

    def __init__(self, lambd, degree, **kwargs):
        """Instantiate a polynomial kernel.

        Parameters
        ----------
        lambd: `float`
            Regularization parameter
        degree: `int`
            Degree of the polynomial
        """
        super().__init__(lambd=lambd, degree=degree, **kwargs)

    @property
    def d(self):
        return self.hyperparams["degree"]

    def eval(self, x, z):
        """Evaluate the kernel on given points. Kenrel action is defined as

            K(x,z) = (1 + x*z)**d

        Parameters
        ----------
        x: `np.array`
            A column vector of features.
        z: `np.array`
            A column vector of ``n`` points at which to evaluate the kernel.

        Returns
        -------
        eval: `np.array`
            An n-by-d matrix where each column is the kernel evaluated at a
            single point.
        """
        return (1 + np.outer(x, z))**self.d


class RBFKernel(Kernel):
    """Class implementing the RBF kernel."""

    def __init__(self, lambd, gamma, **kwargs):
        """Instantiate an RBF kernel.

        Parameters
        ----------
        lambd: `float`
            Regularization parameter
        gamma: `int`
            Degree of the polynomial
        """
        super().__init__(lambd=lambd, gamma=gamma, **kwargs)

    @property
    def gamma(self):
        return self.hyperparams["gamma"]

    def eval(self, x, z):
        """Evaluate the kernel on given points. Kernel action is defined as

            K(x,z) = exp(-gamma*(x-z)^2)

        Parameters
        ----------
        x: `np.array`
            A column vector of features.
        z: `np.array`
            A column vector of ``n`` points at which to evaluate the kernel.

        Returns
        -------
        eval: `np.array`
            An n-by-d matrix where each column is the kernel evaluated at a
            single point.
        """
        return np.exp(-self.gamma * np.subtract.outer(x, z)**2)


class KernelFactory:
    """Kernel factory."""

    @staticmethod
    def create(kernelType, *args, **kwargs):
        """Given either ``poly`` or `rbf`` kernel types and arguments returns
        an instance of PolynomialKernel or RBFKernel.

        Parameters
        ----------
        kernelType: `str`
            A string containing either ``poly`` or ``rbf`` targeting which
            kernel to isntantiate
        args
        kwargs
            All other arguments are passed to the class instntiation call.
        """
        if "poly" in kernelType.lower():
            return PolynomialKernel(*args, **kwargs)
        elif "rbf" in kernelType.lower():
            return RBFKernel(*args, **kwargs)


def truth(x):
    """The truth, the true model that we are trying to reconstruct. A function

        f(x) = 4*sin(pi*x) * cos(6*pi*x^2)

    Parameters
    ----------
    x: `np.array`
        Array of points in which to evaluate the function

    Returns
    -------
    y: `np.array`
        Function values (i.e. labels in this context)
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x**2)


def cross_validate(kernel, x, y, foldSize):
    """Performs cross validation of the kernel on the given dataset by randomly
    permuting the order of features, training on the fold-sized subsets of the
    total dataset {x_i, y_i} and then scoring the predictions of the trained
    kernel.

    Parameters
    ----------
    kernel: `object`
        A Kernel subclass.
    x: `np.array`
        Features
    y: `np.array`
        Labels
    foldSize: `int`
        The size of the selected subsets of data to train on

    Returns
    -------
    meanScore: `float`
        The mean of all of the scores scored after training on random subsets.
    """
    n = len(x)
    idxs = np.random.permutation(np.arange(0, n))

    if foldSize == 0:
        kernel.fit(x, y)
        scores = np.array([kernel.score(x, y)])
    else:
        scores = np.zeros(foldSize)
        for i in range(foldSize):
            lower = int(n/foldSize * i)
            upper = int(n/foldSize * (i+1))

            xPredict = x[idxs[lower:upper]]
            yPredict = y[idxs[lower:upper]]
            xFit = np.concatenate([x[idxs[0:lower]], x[idxs[upper:]]])
            yFit = np.concatenate([y[idxs[0:lower]], y[idxs[upper:]]])

            kernel.fit(xFit, yFit)
            scores[i] = kernel.score(xPredict, yPredict)

    return np.mean(scores)


def sampler(x, y, lambdas, hyperparams, foldSize, kernelType):
    """Preforms grid sampling of cross evaluated scores over all pairs of
    given lambdas and hyperparameters.

    Parameters
    ----------
    x: `np.array`
        Features
    y: `np.array`
        Labels
    lambdas: `np.array`
        Regularization parameters at which to preform cross validation.
    hyperparams: `np.array`
        Hyperparameter values at which to preform cross validation.
    foldSize: `int`
        Cross validation folding size.
    kernelType: `str`
        Which kernel type to use.

    Returns
    -------
    samples: `np.array`
        A structured numpy array containing the cross validation error, log
        of regularization parameters and hyperparameter values at which the
        error was measured.
    bestFit: `dict`
        A dictionary containing the minimal sampled cross validation error and
        the values of lambda and hyperparameter at that error.
    """
    nCombinations = len(lambdas)*len(hyperparams)
    dt = [("lambda", float), ("hyperparams", float), ("error", float)]
    samples = np.zeros((nCombinations, ), dtype=dt)

    for i, (lambd, hparam) in enumerate(itertools.product(lambdas, hyperparams)):
        k = KernelFactory.create(kernelType, lambd, hparam)
        samples["error"][i] = cross_validate(k, x, y, foldSize)
        samples["lambda"][i] = lambd
        samples["hyperparams"][i] = hparam

    idxMinErr = np.where(samples['error'] == samples['error'].min())[0][0]
    minErr = samples['error'][idxMinErr]
    optimLambda = samples['lambda'][idxMinErr]
    optimHParam = samples['hyperparams'][idxMinErr]
    print(f"Optimal lambda: {optimLambda:.4}, hyperparam: {optimHParam:.4} sampled "
          f"@ minimal error: {minErr:.4} (log(err)={np.log(minErr):.4}).")

    return samples, {"minErr": minErr, "lambda": optimLambda,
                     "hyperparam": optimHParam}


def plotA3a(fig, ax, x, y, z, bestFit, xlabel="Hyperparameter",
            ylabel="log(lambda)", cbarlbl="log(error)", title="Kernel"):
    """Plots the grid sampled cross validation errors as a function of sampled
    regularization and hyper-parameters.


    Parameters
    ----------
    fig: `matplotlib.pyplot.Figure`
       Figure to which a colorbar will be attached.
    ax: `matplotlib.pyplot.Axes`
       Axis on which to plot
    x: `np.array`
        Features, a vector of length n
    y: `np.array`
        Labels, a vector of length m
    z: `np.array`
        An m*n lenght array of cross-validation errors.
    bestFit: `dict`
        dictionary containing the minimal error, and the values of
        regularization and hyper-parameters at that point.
    xlabel: `str`, optional
        X axis label. Defaults to hyperparameter
    ylabel: `str`, opitonal
        Y axis label. Defaults to log(lambda)
    cbarlbl: `str`, optional
        Colorbar label. Defaults to log(error)
    title: `str`, optional
        Axis title. Defaults to "Kernel".

    Returns
    -------
    ax: `matplotlib.pyplot.Axes`
        Modified axis containing the plot.
    cbar: `matplotlib.pyplot.Axes`
        Axis containing the colorbar.
    """
    smpls = z.reshape(len(y), len(x))

    img = ax.imshow(smpls, interpolation=None, aspect="auto",
                    extent=(x[0], x[-1], y[0], y[-1]), origin="lower")
    cbar = fig.colorbar(img, ax=ax, orientation='vertical')
    ax.axvline(bestFit['hyperparam'], color="white")
    ax.axhline(bestFit['lambda'], color="white")

    cbar.set_label(cbarlbl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return ax, cbar


def A3a(kernelType, x, y, n, foldSize, lambdas, hyperparams, xlabel="degree",
        title="Polynomial Kernel"):
    """Samples the cross validation error on a grid for both polynomial and RBF
    kernels. Reports the minimal found error values and plots the errors.

    Parameters
    ----------
    foldSize: `int`
        Cross validationsubset size
    lambdas: `np.array`, optional
        Regularization parameters at which to calculate cross validation error.
        Default: np.logspace(10, -13, 50).
    plot: `bool`
        Display plots.
    """
    samples, best = sampler(x, y, lambdas, hyperparams, foldSize=foldSize,
                            kernelType=kernelType)

    fig, axes = plt.subplots()
    plotA3a(fig, axes, hyperparams, lambdas, np.log(samples['error']),
            best, xlabel=xlabel, title=title)

    return samples, best


def A3b(kernelType, x, y, bestFit):    
    kernel = KernelFactory.create(kernelType, bestFit['lambda'], bestFit['hyperparam'])
    kernel.fit(x, y)
    
    fig, axes = plt.subplots(1, 2)

    xTest = np.linspace(x.min(), x.max(), 100)
    yHat = kernel.predict(xTest)

    axes[0].scatter(x, y, label="Data")
    axes[0].plot(xTest, yHat, label="Kernel Regression")
    axes[0].plot(xTest, truth(xTest), label="True f(x)")
    kernel.plot_basis(axes[1], xTest)

    axes[0].set_title("Polynomial Kernel n = 30")
    axes[0].legend()
    axes[1].set_title("Kernel Basis functions")

    return fig, axes


def bootstrap(x, y, B, x_test, predictor, *predictor_args):
    n = x.shape[0]
    predictions = np.zeros((B, x_test.shape[0]))
    
    for i in range(B):
        bootstrap_idxs = np.random.choice(np.arange(n), size=n)
        x_bootstrap = x[bootstrap_idxs]
        y_bootstrap = y[bootstrap_idxs]

        predictor.train(x_bootstrap, y_bootstrap, *predictor_args)
        predictions[i] = predictor.predict(x_test)

    return predictions


def A3(doPoly=True, doRBF=True):
    n, foldSize = 30, 30
    x = np.random.uniform(size=n)
    y = truth(x) + np.random.normal(size=n)

    # A.3 for poly
    if doPoly:
        lambdas = np.linspace(0.4, 1, 100)
        degrees = np.arange(30, 60, 1)
        samples, bestFit = A3a("poly", x, y, n, foldSize, lambdas, degrees)
        A3b("poly", x, y, bestFit)

        x_test = np.linspace(0, 1, 100)
        kernel = KernelFactory.create(kernelType, bestFit['lambda'], bestFit['hyperparam'])
        bootstrap(x, y, x_test, kernel, )

    # A.3 for RBF
    if doRBF:
        lambdas = np.linspace(0.0001, 0.1, 50)
        gammas = np.linspace(120, 160, 60)
        samples, bestFit = A3a("rbf", x, y, n, foldSize, lambdas, gammas,
                               xlabel="gamma", title="RBF Kernel")
        A3b("rbf", x, y, bestFit)

    plt.show()


x_test = np.linspace(0, 1, 100)
B = 300

poly_predictions = bootstrap(x_30, y_30, B, x_test, Kernel("poly"), best_predictor_arg_poly_30, best_lambda_poly_30)
rbf_predictions = bootstrap(x_30, y_30, B, x_test, Kernel("rbf"), best_predictor_arg_rbf_30, best_lambda_rbf_30)


def percentile_plot(x, y, predictions, title):
    x_test = np.linspace(0, 1, 100)
    
    plt.scatter(x, y, alpha=0.3, label="Data")
    
    # Mean prediction
    plt.plot(x_test, np.mean(predictions, axis=0), color='orange', lw=2, label="Kernel Regression")

    # Confidence Intervals
    plt.fill_between(x_test, np.percentile(predictions, 5, axis=0), 
                 np.percentile(predictions, 95, axis=0), color='orange', alpha=0.3)
    plt.plot(x_test, np.percentile(predictions, 95, axis=0), color='orange', ls='-')
    plt.plot(x_test, np.percentile(predictions, 5, axis=0), color='orange', ls='-')

    # True f(x)
    plt.plot(x_test, f(x_test), color='blue', ls='--', lw=2, label="True f(x)")
    
    plt.title(title)
    
    plt.ylim((y.min() - 1, y.max() + 1))
    
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    A3()
