import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
from torch import nn, optim
from torch.nn import functional
import torch.utils.data as datutils

from torchvision import datasets, transforms


torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_mnist_dataset(path="data/mnist_data/", digit=None, batchSize=1):
    """Loads MNIST data located at path.

    MNIST data are 28x28 pixel large images of numbers.

    Parameters
    ----------
    path : `str`
        Path to the data directory
    digit : `int` or `None`, optional
        Load only data for a specific digit.
    batchSize: `int`, optional
        Batch size.

    Returns
    -------
    train : `torch.DataLoader`
        A generator that will shuffle the data at every iteration, yields train
        datasets
    trainlabels : `torch.DataLoader`
        A generator that returns the test dataset.

    Notes
    -----
    Data are normalized upon loading to the mean of the dataset.
    Data are downloaded if not present at path.
    """
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(path, train=True, download=True, transform=trans)
    test = datasets.MNIST(path, train=False, transform=trans)

    trainMask, testMask = np.arange(len(train)), np.arange(len(test))
    if digit is not None:
        trainMask = np.where(train.targets == digit)[0]
        testMask = np.where(test.targets == digit)[0]

    maskTrain = datutils.Subset(train, trainMask)
    maskTest = datutils.Subset(test, testMask)
    trainLoader = datutils.DataLoader(maskTrain, batch_size=batchSize, shuffle=True)
    testLoader = datutils.DataLoader(maskTest, batch_size=batchSize, shuffle=False)

    return trainLoader, testLoader


class MNISTAutoencoder(nn.Module):
    """MNIST Autoencoder Module.

    Parameters
    ----------
    encode: `torch.nn.Sequential`
        Encoding steps
    decode: `torch.nn.Sequential`
        Decoding steps
    imgSize: `int`, optional
        The image dimensions multiplied. Defaults to 28*28
    """
    def __init__(self, encode, decode, *args, **kwargs):
        super().__init__()
        self.encode = encode
        self.decode = decode
        self.imgSize = kwargs.pop("imgSize", 28*28)

    def forward(self, x):
        """Encode the given data and then decode it.

        Parameters
        -----------
        x : `torch.tensor`
            Data to encode

        Returns
        -------
        decoded : `torch.tensor`
            The decoded encoded data.
        """
        z = self.encode(x.view(-1, self.imgSize))
        return self.decode(z)

    def loss(self, x):
        """Given the data, alculate MSE loss of the reconstruction. Ensure the
        autoencoder has been trained.

        Parameters
        -----------
        x : `torch.tensor`
            Data 

        Returns
        -------
        loss : `nn.MSELoss`
            MSE Loss
        """
        reconstructed = self.forward(x)
        loss = functional.mse_loss(reconstructed, x.view(-1, self.imgSize))
        return loss


class A3aAutoencoder(MNISTAutoencoder):
    """Simple single linear encode and decode layer autoencoder.

    Parameters
    ----------
    imgSize : `int`, optional
        Size of the images. Default 28*28
    hiddenSize : `int`, optional
        Size of the encoding layer. Default 64
    """
    def __init__(self, imgSize=28*28, hiddenSize=64, **kwargs):
        super().__init__(
            encode = nn.Sequential(nn.Linear(imgSize, hiddenSize)),
            decode = nn.Sequential(nn.Linear(hiddenSize, imgSize)),
            imgSize = imgSize,
            **kwargs
        )
        self.hiddenSize = hiddenSize


class A3bAutoencoder(MNISTAutoencoder):
    """Simple single non-linear (ReLU) encode and decode layer autoencoder.

    Parameters
    ----------
    imgSize : `int`, optional
        Size of the images. Default 28*28
    hiddenSize : `int`, optional
        Size of the encoding layer. Default 64
    """

    def __init__(self, imgSize=28*28, hiddenSize=64, **kwargs):
        super().__init__(
            encode = nn.Sequential( nn.Linear(imgSize, hiddenSize), nn.ReLU() ),
            decode = nn.Sequential( nn.Linear(hiddenSize, imgSize), nn.ReLU() ),
            imgSize = imgSize,
            **kwargs
        )
        self.hiddenSize = hiddenSize
        

def train(dataLoader, model, optimizer, epoch, verbosity=20):
    """Trains the model using the given batched data. Calculates loss for each
    batch as well as the total average loss across the epoch and prints them.

    Parameters
    ----------
    dataLoader: `torch.DataLoader`
        Generator that returnes batched train data.
    model : `obj`
        One of the Autoencoders or some other nn.Module with a loss method.
    optimizer : `obj`
        One of pytorches optimizers (f.e. torch.optim.Adam)
    epoch : `int`
        What epoch is this training step performing, used to calculate loss
    verbosity: `int`
        How often are batch losses printed, larger number means less prints.
        Epoch average loss is always printed.
    """
    # DataLoader length is number of batches that fit in the dataset.
    # Length of the dataset is the actual number of data points
    # used (f.e. the total number of CIFAR images)
    nAll, nBatches = len(dataLoader.dataset), len(dataLoader)
    verbosity = 1 if nBatches/verbosity < 1 else np.ceil(nBatches/verbosity)
    trainLoss, avgTrainLoss = 0, 0
    print(f"Epoch: {epoch}:")
    for i, (data, labels) in enumerate(dataLoader):
        data = data.to(DEVICE)

        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        optimizer.step()

        lss = loss.item()
        trainLoss += lss
        avgTrainLoss += lss
        if i % verbosity == 0 and i!=0:
            # The length of data, loaded by loader, is at most the batch size,
            # not neccessarily equal for all batches (i.e. last one might be shorter).
            nBatch = len(data)
            print(
                f"    [{i*nBatch:>6}/{nAll:>6} ({100.0*i/nBatches:<5.4}%)]"
                f"    Loss: {trainLoss/verbosity:>10.8f}"
            )
            trainLoss = 0.0

    print(f"    Avg train loss: {avgTrainLoss/nBatches:>15.4f}")


def test(dataLoader, model, optimizer):
    """Calculate and print test losses.

    Parameters
    ----------
    dataLoader: `torch.DataLoader`
        Generator that returnes batched train data.
    model : `obj`
        One of the Autoencoders or some other nn.Module with a loss method.
    optimizer : `obj`
        One of pytorches optimizers (f.e. torch.optim.Adam)
    """
    testLoss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(dataLoader):
            data = data.to(DEVICE)
            testLoss += model.loss(data).item()
    testLoss /= len(dataLoader.dataset)
    print(f"Test set loss: {testLoss:.8f}\n")


def learn(trainDataLoader, testDataLoader, model, epochs, learningRate=1e-3,
          verbosity=20):
    """Trains a model for n epochs using Adam optimizer, and then estimates
    test dataset losses.

    Parameters
    ---------
    trainDataLoader: `torch.DataLoader`
        Generator that returnes batched train data.
    testDataLoader: `torch.DataLoader`
        Generator that returnes batched test data.
    model : `obj`
        One of the Autoencoders or some other nn.Module with a loss method.
    epochs : `int`
        Number of epochs to train for.
    learningRate : `float`, optional
        Learning rate of the Adam optimizer. Default: 0.001
    verbosity: `int`
        How often are batch losses printed, larger number means less prints.
        Epoch average loss is always printed. Default: 20
    """
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    for epoch in range(1, epochs + 1):
        train(trainDataLoader, model, optimizer, epoch, verbosity=verbosity)
        test(testDataLoader, model, optimizer)


def showimages(images, interpolation="nearest"):
    """Given a list of images (np.array or torch.tensor) creates a figure with
    sufficient number of columns and plots each image in a separate one.

    Parameters
    ----------
    images : `list` or `tuple`
        List of images to plot

    Returns
    -------
    fig : `matplotlib.pyplot.Figure`
        Figure
    axes : `matplotlib.pyplot.Axes`
        Axes
    """
    fig, axes = plt.subplots(1, len(images), figsize=(10, 3), sharex=True, sharey=True)
    axes = np.array([axes]) if len(images) == 1 else axes
    for img, ax in zip(images, axes.ravel()):
        ax.imshow(img.cpu(), interpolation=interpolation)
        ax.set_axis_off()
    fig.set_tight_layout(True)
    return fig, ax


def plot_reconstructions(dataLoader, model):
    """Given a trained model reconstructs a single example of each of the unique
    labels present in the data and plots them side by side.

    Parameters
    ----------
    dataLoader : `torch.DataLoader`
        Generator that returns batched data
    model : `obj`
        One of the Autoencoders, or some other nn.Module.
    """
    # get unique digits in the dataset via this monstrosity. First dataset is
    # the subset, second dataset is the whole dataset, the targets are the
    # labels of that dataset. We can only plot for the digits we trained for,
    # so sample unique digits from the subset. Convert to numpy to get non-
    # tensor values and finally list gets us .pop()
    allData = dataLoader.dataset.dataset
    subsetIdxs = dataLoader.dataset.indices
    digits = list(allData.targets[subsetIdxs].unique().numpy())
    stacks = []
    for data, label in allData:
        data = data.to(DEVICE)
        if label == digits[0]:
            with torch.no_grad():
                reconstruction = model(data)
            stack = torch.cat([data[0], reconstruction.view(28, 28)])
            stacks.append(stack)
            digits.pop(0)
        if len(digits) == 0:
            break
    return showimages(stacks)


def A3a(epochs=5, learningRate=1e-3, batchSize=128, verbosity=5):
    """Loads MNIST train and test datasets, batches them, trains a single layer
    linear Autoencoder, reports on the train and test losses and plots the
    reconstructons.

    Parameters
    ----------
    epochs : `int`, optional
       Number of training epochs. Default: 
    learningRate : `float`, optional
       Optimizers learning rate Default: 1e-3.
    batchSize : `int`, optional
        Size of the batched dataset. Default: 128
    """
    trainLoader, testLoader = load_mnist_dataset(batchSize=batchSize)

    for latentSize in (32, 64, 128):
        AC = A3aAutoencoder(latentSize=latentSize).to(DEVICE)
        learn(trainLoader, testLoader, AC, epochs, learningRate, verbosity)
        fig, axes = plot_reconstructions(trainLoader, AC)
        fig.suptitle("Digit reconstruction, linear \n"
                     f"(h={latentSize}, Nbatch={batchSize}, epochs={epochs}, "
                     f"lr={learningRate})")
        fig.savefig(f"../HW4_plots/A3a_h{latentSize}.png")
    plt.show()


def A3b(epochs=5, learningRate=1e-3, batchSize=128, verbosity=5):
    """Loads MNIST train and test datasets, batches them, trains a single layer
    non-linear Autoencoder, reports on the train and test losses and plots the
    reconstructons.

    Parameters
    ----------
    epochs : `int`, optional
       Number of training epochs. Default: 
    learningRate : `float`, optional
       Optimizers learning rate Default: 1e-3.
    batchSize : `int`, optional
        Size of the batched dataset. Default: 128
    """
    trainLoader, testLoader = load_mnist_dataset(batchSize=batchSize)

    for latentSize in (32, 64, 128):
        AC = A3bAutoencoder(latentSize=latentSize).to(DEVICE)
        learn(trainLoader, testLoader, AC, epochs, learningRate, verbosity)
        fig, axes = plot_reconstructions(trainLoader, AC)
        fig.suptitle("Digit reconstruction, non-linear \n"
                     f"(h={latentSize}, Nbatch={batchSize}, epochs={epochs}, "
                     f"lr={learningRate})")
        fig.savefig(f"../HW4_plots/A3b_h{latentSize}.png")
    plt.show()


def A3():
    """Runs part a and b of problem A3 in HW4"""
    A3a()
    A3b()


if __name__ == "__main__":
    A3()
    pass








class MyAutoencoder(MNISTAutoencoder):
    def __init__(self, imgSize=28*28, hiddenSize=128, latSize=2, **kwargs):
        super().__init__(
            encode = nn.Sequential(
                nn.Linear(imgSize, hiddenSize),
                nn.ReLU(),
                nn.Linear(hiddenSize, latSize)
            ),
            decode = nn.Sequential(
                nn.Linear(latSize, hiddenSize),
                nn.ReLU(),
                nn.Linear(hiddenSize, imgSize)
            ),
            imgSize = imgSize,
            hiddenSize = hiddenSize,
            latentSize = latSize,
            **kwargs
        )

    def forward_mu(self, x):
        """Get the latent space layer back."""
        return self.encode(x.view(-1, self.imgSize))



def plot_latent(dataLoader, model, vectors=(0,1)):
    """For 2D latent spaces 
    """

    digits = range(10)
    colors = [cm.tab10(digit/10.0) for digit in digits]
    coldict = {d:c for d, c in zip(digits, colors)}

    fig, axes = plt.subplots()

    # fake legend
    for d, c in zip(digits, colors):
        plt.scatter(-100, -100, c=[c], label=d)
    plt.legend()

    # get the latent space values and plot them in colors of digits
    for i, (data, labels) in enumerate(dataLoader):
        with torch.no_grad():
            data = data.to(DEVICE)
            mu = model.forward_mu(data.view(-1, 784))
            for digit, color in zip(digits, colors):
                digitMask = labels == digit
                x, y = vectors
                plt.scatter(mu[:, x][digitMask].cpu(), mu[:, y][digitMask].cpu(),
                            c=[color], alpha=0.12)

    plt.title("Autoencoders Latent space")
    plt.xlabel(f"Dimension {vectors[0]}")
    plt.ylabel(f"Dimension {vectors[1]}")
    plt.xlim(-15, 15)
    plt.ylim(-5, 5)
    plt.tight_layout()
    return fig, axes



def extra_autoencoder(epochs=10, learningRate=1e-3, batchSize=1024):
    trainLoader, testLoader = load_mnist_dataset(batchSize=batchSize)

    hiddenDim, latDim = 400, 3
    AC = MyAutoencoder(hiddenSize=hiddenDim, latSize=latDim).to(DEVICE)
    learn(trainLoader, testLoader, AC, epochs, learningRate)
    fig, axes = plot_reconstructions(trainLoader, AC)
    fig.suptitle("Digit reconstruction, non-linear (my autoencoder) \n"
                 f"(h={hiddenDim}, latDim={latDim}, Nbatch={batchSize}, epochs={epochs}, "
                 f"lr={learningRate})")
    plt.show()

    fig, axes = plot_latent(trainLoader, AC)
    fig.savefig("../HW4_plots/MyAutoencoder_LatentSpace1.png")
    plt.show()
    fig, axes = plot_latent(trainLoader, AC, (0, 2))
    fig.savefig("../HW4_plots/MyAutoencoder_LatentSpace2.png")
    plt.show()
    fig, axes = plot_latent(trainLoader, AC, (1, 2))
    fig.savefig("../HW4_plots/MyAutoencoder_LatentSpace3.png")
    plt.show()


extra_autoencoder()
