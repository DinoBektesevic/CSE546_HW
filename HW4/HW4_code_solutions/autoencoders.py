import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional
import torch.utils.data as datutils

from torchvision import datasets, transforms
from torchvision.utils import make_grid


batch_size   = 128  # input batch size for training
epochs       = 10   # number of epochs to train
seed         = 1    # random seed
lr           = 1e-3 # learning rate
img_size     = 28*28 # MNIST images are 28x28


torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_mnist_dataset(path="data/mnist_data/", digit=None):
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
    trainLoader = datutils.DataLoader(maskTrain, batch_size=batch_size, shuffle=True)
    testLoader = datutils.DataLoader(maskTest, batch_size=batch_size, shuffle=False)

    return trainLoader, testLoader


train_loader, test_loader = load_mnist_dataset()


class Autoencoder(nn.Module):
    def __init__(self, med_size=400):
        super().__init__()
        self.en1 = nn.Linear(img_size, med_size)
        self.de1 = nn.Linear(med_size, img_size)

    def encode(self, x):
        return functional.relu(self.en1(x))

    def decode(self, z):
        return functional.relu(self.de1(z))

    def forward(self, x):
        z = self.encode(x.view(-1, img_size))
        return self.decode(z)
    
    def loss(self, x):
        reconstructed = self.forward(x)
        loss = functional.mse_loss(reconstructed, x.view(-1, img_size))
        return loss


def train(model, optimizer, epoch):
    trainLoss = 0
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        trainLoss += loss.item()
        optimizer.step()

        if i % 10 == 0:
            # data loader length is number of batches that fit in the dataset.
            # The length of data, loaded by loader, is at most the batch size
            # and the length of the dataset is the actual number of data points
            # used (f.e. the total number mnist images of the same digit)
            nBatches = len(train_loader)
            nBatch = len(data)
            nAll = len(train_loader.dataset)
            print(f'    Train Epoch: {epoch} [{i*nBatch}/{nAll} '
                  f'({100* i/nBatches:.04}%)]        Loss: {loss.item()/nBatch:.6f}')

    print('Epoch: {} Average loss: {:.4f}'.format(
          epoch, trainLoss / len(train_loader.dataset)))


def test(model, optimizer, epoch):
    testLoss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            reconBatch = model(data)[0]
            testLoss += model.loss(data).item()

    testLoss /= len(test_loader.dataset)
    print('Test set loss: {:.4f}'.format(testLoss))


def learn(model, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        train(model, optimizer, epoch)
        test(model, optimizer, epoch)


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.axis("off")
    plt.show()


def show_recon(model, n=9):
    with torch.no_grad():
        for i, (data, labels) in enumerate(train_loader):
            if i == 0:
                # this is a call to forward, that returns reconstructed image,
                # mu and the variance
                batchReconstructions = model(data)
                comparison = torch.cat([data[:n],
                                      batchReconstructions.view(batch_size, 1, 28, 28)[:n]])
                show(make_grid(comparison, nrow=n))


AC = Autoencoder().to(device)
learn(AC, 10)
show_recon(AC)
plt.show()
