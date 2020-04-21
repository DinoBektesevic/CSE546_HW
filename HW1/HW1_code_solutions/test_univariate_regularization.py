"""
    TEST SCRIPT FOR POLYNOMIAL REGRESSION REGULARIZATION PARAMETER
    AUTHOR Dino Bektesevic
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from polyreg import PolynomialRegression

if __name__ == "__main__":
    """Test effects of regularization parameters by plotting
    results of fits with different regularization parameters.
    """
    # pick the polynomial degree and regularization parameters
    d = 8
    reg_lambdas = list(range(0, 50, 8))
    reg_lambdas.extend([100, 1000, 10000])

    # load the data, accidentaly override a python builtin keyword
    filePath = "data/polydata.dat"
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    # adopt a horrible naming convention
    X = allData[:, [0]]
    y = allData[:, [1]]

    # fit with different regression parameters and store results
    xAxisData, yAxisData = [], []
    for reg_lambda in reg_lambdas:
        model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
        model.fit(X, y)

        # output predictions
        xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
        ypoints = model.predict(xpoints)

        xAxisData.append(xpoints)
        yAxisData.append(ypoints)

    # overplot fit results
    colors = cm.viridis(np.linspace(0, 1, len(yAxisData)))
    fig, ax = plt.subplots()
    # points
    ax.plot(X, y, 'rx')
    for x, y, color, label in zip(xAxisData, yAxisData, colors, reg_lambdas):
        ax.plot(x, y, color=color, label=f"Reg. param: {label}")
    ax.set_title('PolyRegression with d = '+str(d))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.legend()
    plt.show()
