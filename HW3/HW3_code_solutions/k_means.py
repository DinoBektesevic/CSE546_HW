import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

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
    train : `np.array`
        train data normalized to 1
    trainlabels : `np.array`
        train data labels
    test : `np.array`
        test data normalized to 1
    testLabels : `np.array`
        test data labels
    """
    mndata = MNIST(path)

    train, trainLabels = map(np.array, mndata.load_training())
    test, testLabels  = map(np.array, mndata.load_testing())

    train = train/255.0
    test = test/255.0

    return train, trainLabels, test, testLabels


def k_means_objective(clusters, centers):
    """Calculates the sum of distances of points in a cluster to the given
    cluster centers. This is the k-means objective, defined as:

        F(mu, C) = \sum^m || mu_j - x_j ||^2

    where mu are the cluster centers and x_j the point in that cluster.

    Parameters
    ----------
    clusters: `np.array`
        Clusters of points (each element a collection of points belonging to
        that cluster)
    centers: `np.array`
        Coordinates of centers of corresponding clusters.

    Returns
    -------
    objective: `float`
        K-means objective
    """
    dist = [np.sum(np.linalg.norm(c-p, axis=1)) for c, p in zip(centers, clusters)]
    return np.sum(dist)


def cluster_data(points, centers):
    """Calculates distance from each point to all given clusters and forms
    clusters by associatin points with its closest center.

    Parameters
    ----------
    points: `np.array`
        Points to cluster
    centers: `np.array`
        Cluster centers.

    Returns
    -------
    clusters: `np.array`
        Array each element of which is an cluster. A cluster is an array of
        points that bellonging to the same cluster.
    """
    nClusters = len(centers)
    # equivalent to: np.linalg.norm(points - centers[:, None], axis=2)
    # but for more clusters memory just runs out
    distances = [np.linalg.norm(points - center, axis=1) for center in centers]
    closestClusters = np.argmin(np.array(distances), axis=0)
    clusters = np.array([points[closestClusters == i] for i in range(nClusters)])
    return clusters


def calculate_centers(clusters):
    """Calculates centers of given clusters by calculating the mean of the
    individual coordinates of points in that cluster.

    Parameters
    ---------
    clusters: `np.array`
        An array in which each element is an array of points belonging to the
        cluster.

    Returns
    -------
    centers: `np.array`
        Cluster centers.
    """
    cluster_sizes = np.array([cluster.shape[0] for cluster in clusters])
    centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])
    return centers


def loyds_alg(points, nClusters, tolerance=0.01, nIter=None):
    """Iteratively calculates centers and re-assinged clusters based on those
    centers until convergence is achieved. Intial centers are selected as
    random points from the given dataset. Convergence is achieved then the new
    center coordinate components maximal distance from the old center
    coordinates is smaller than tolerance. This is known as Loyd's algorithm
    for calculating k-means.

    Paramaters
    ----------
    points: `np.array`
        Array of points to cluster
    nClusters: `int`
        Number of clusters
    tolerance: `float`, optional
        If new centers move, cumulatively, by less than tolerance the iteration
        is terminated. Unless a specific number of iterations was given. By
        default: 0.01
    nIter: `int`, optional
        If given, tolerance is ignored and iterations are carried out for the
        given number of iterations. 
    """
    # assign first centers to be random points in the dataset
    centers = points[np.random.permutation(np.arange(len(points)))[:nClusters]]
    clusters = cluster_data(points, centers)

    oldCenters, oldClusters = centers, clusters
    objectives = []
    converged, drOld, i = False, None, 0
    while not converged:
        centers = calculate_centers(clusters)
        clusters = cluster_data(points, centers)

        # worlds most complicated break-out logic
        dr = np.linalg.norm(centers - oldCenters)
        if drOld is None:
            print(dr)
            drOld = dr
        elif np.abs(drOld-dr) < tolerance:
            print(dr, "         ", drOld-dr)
            if nIter is not None:
                if i >= nIter:
                    converged = True
            else:
                converged = True
        else:
            print(dr, "         ", drOld-dr)
            drOld = dr
        
        objective = k_means_objective(clusters, centers)
        objectives.append(objective)
        
        oldCenters = centers
        oldClusters = clusters
        i+=1

    return centers, clusters, objectives


def plot_objectives(ax, objectives, xlabel="Iteration number", ylabel="Objective",
                    title="Cluster centers"):
    """Plots objectives.

    Paramters
    ---------
    ax: `matplotlib.pyplot.Axes`
        Axis to plot on.
    objectives: `np.array`
        Array of objective scores.
    xlabel: `str`, optional
        X label
    ylabel: `str`, optional
        Y label
    title: `str`, optional
        Axis title.
    """
    ax.plot(objectives)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def plot_centers(axes, centers, title=""):
    """Plots centers as 28x28 images.

    Paramters
    ---------
    axes: `matplotlib.pyplot.Axes`
        Axes to plot on.
    centers: `np.array`
        Centers to plot.
    title: `str`, optional
        Axis title.
    """
    for ax, center in zip(axes.ravel(), centers):
        ax.imshow(center.reshape((28, 28)), cmap='gray')
        ax.set_title(title)
        plt.axis("off")
    return axes


def A4b(k=10, tolerance=0.01, nIter=100):
    """Runs Loyd's k-means algorithm on the MNIST test data-set for 100
    iterations, clustering the data into 10 clusters, and plots the objective
    function vs iteration number and the found centers.

    Parameters
    ----------
    k: `int`
        Number of clusters
    tolerance: `float`, optional
        Convergence tolerance, ignored if nIter is given.
    nIter: `int`, optional
        Number of iterations to carry out, defaults to 100.
    """
    test, testLabels, train, trainLabels = load_mnist_dataset()
    centers, clusters, objectives = loyds_alg(test, k, tolerance, nIter)

    fig1, axis1 = plt.subplots()
    plot_objectives(axis1, objectives)
    plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(10, 25), sharex=True, sharey=True)
    plot_centers(axes, centers)
    plt.axis("off")
    plt.show()


def error(clusters, centers, nPoints=1):
    """Given clusters and centers calculates the total distance of points in
    that cluster to its center ("spread of points") and adds the least spread
    out cluster distances together into a total distance. Normalizes the total
    distance by the number of points.

    Paramters
    ---------
    clusters: `np.array`
        Array of clusters of points.
    centers: `np.array`
        Array of centers of clusters.
    nPoints: `int`, optional
        Normalization factor. Defaults to 1.

    Returns
    -------
    totDist: `float`
        Total sum of "spreads" of all the points in their respective clusters.
    """
    totDist = 0
    for center in centers:
        allDists = []
        for cluster in clusters:
            dr = np.linalg.norm(cluster-center, axis=1)
            allDists.append(np.sum(dr))
        totDist += min(allDists)
    return totDist/nPoints


def A4c(k=(2, 4, 6, 8, 16, 32, 64)):
    """Clusters MNIST test and train datasets into 2^n n=1,2,3,4,5,6 clusters
    and calculates the total error of the clustering as a function of the
    number of clusters. Plots the error.
    Clustering iterations are terminated when the total moved center distances
    are less than 0.01.

    Parameters
    ----------
    k: `tuple`
        Tuple of integers representing number of iterations.
    """
    test, testLabels, train, trainLabels = load_mnist_dataset()

    testErrors, trainErrors = [], []
    for nClusters in k:
        centers, clusters, objectives = loyds_alg(test, nClusters)
        testErrors.append(error(clusters, centers, len(test)))
        testClusters = cluster_data(train, centers)
        trainErrors.append(error(testClusters, centers, len(train)))

    plt.plot(k, testErrors, label="test errors")
    plt.plot(k, trainErrors, label="train errors")
    plt.xlabel("N. Clusters (k)")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    A4b()
    A4c()
