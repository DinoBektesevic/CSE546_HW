import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
from torch import nn, optim
from torch.nn import functional
import torch.utils.data as datutils

from torchvision import datasets, transforms
from torchvision.utils import make_grid


torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_cifar_dataset(path="data/cifar_data/", pickClass=None, batchSize=1, 
                       validationFrac=0.2):
    """Loads CIFAR data located at path.

    CIFAR data are 28x28 pixel large images of numbers.

    Parameters
    ----------
    path : `str`
        Path to the data directory
    digit : `int` or `None`, optional
        Load only data for a specific digit.
    batchSize: `int`, optional
        Batch size. Default: 1
    validationFrac : `float`, optional
        Fraction of train data that will be separated into validation
        dataset.

    Returns
    -------
    trainLoader : `torch.DataLoader`
        A generator that will shuffle and batch the data at every iteration, 
        yields train datasets
    validationLoader : `torch.DataLoader`
        Generator that returns the whole validation dataset.
    trainLoader : `torch.DataLoader`
        Generator that returns the whole test dataset.

    Notes
    -----
    Data are normalized upon loading to the mean of the dataset.
    Data are downloaded if not present at path.
    """
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train = datasets.CIFAR10(path, train=True, download=True, transform=trans)
    test = datasets.CIFAR10(path, train=False, transform=trans)

    trainMask, testMask = np.arange(len(train)), np.arange(len(test))
    if pickClass is not None:
        if type(pickClass) == str:
            pickClass = train.classes.index(pickClass)
        trainMask = np.where(train.targets == pickClass)[0]
        testMask = np.where(test.targets == pickClass)[0]

    maskedTrain = datutils.Subset(train, trainMask)
    maskedTest = datutils.Subset(test, testMask)

    trainLen = int((1-validationFrac)*len(maskedTrain))
    validLen = int(validationFrac*len(maskedTrain))
    maskedTrain, maskedValidation = datutils.random_split(maskedTrain,
                                                          [trainLen, validLen])

    trainLoader = datutils.DataLoader(maskedTrain, batch_size=batchSize, shuffle=True)
    validationLoader = datutils.DataLoader(maskedValidation, batch_size=batchSize, shuffle=False)
    testLoader = datutils.DataLoader(maskedTest, batch_size=batchSize, shuffle=False)

    return trainLoader, validationLoader, testLoader


def train(dataLoader, model, optimizer, epoch, verbosity=5, validationLoader=None):
    """Trains an epoch of the model using the given batched data. Calculates
    loss for each batch as well as the total average loss across the epoch
    and prints them.

    Parameters
    ----------
    dataLoader: `torch.DataLoader`
        Generator that returns batched train data.
    model : `obj`
        One of the Autoencoders or some other nn.Module with a loss method.
    optimizer : `obj`
        One of pytorches optimizers (f.e. torch.optim.Adam)
    epoch : `int`
        What epoch is this training step performing, used to calculate loss
    verbosity: `int`
        How often are batch losses printed, larger number means less prints.
        Epoch average loss is always printed.
    validationLoader : `torch.DataLoader`
        Generator that returns batched validation data.
    """
    # DataLoader length is number of batches that fit in the dataset.
    # Length of the dataset is the actual number of data points
    # used (f.e. the total number of CIFAR images)
    nAll, nBatches = len(dataLoader.dataset), len(dataLoader)
    verbosity = 1 if nBatches/verbosity < 1 else np.ceil(nBatches/verbosity)
    trainLoss, avgTrainLoss, trainAcc, avgTrainAcc = 0, 0, 0, 0
    print(f"Epoch: {epoch}:")
    for i, (data, labels) in enumerate(dataLoader):
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        # allow for multi-GPU via DataParallel
        try:
            loss = model.loss(data, labels)
        except AttributeError:
            loss = model.module.loss(data, labels)
        loss.backward()
        optimizer.step()
        
        acc = model.accuracy(data, labels)
        lss = loss.item()
        trainAcc += acc
        avgTrainAcc += acc
        trainLoss += lss
        avgTrainLoss += lss
        if i % verbosity == 0 and i!=0:
            # The length of data, loaded by loader, is at most the batch size,
            # not neccessarily equal for all batches (i.e. last one might be shorter).
            nBatch = len(data)
            print(
                f"    [{i*nBatch:>6}/{nAll:>6} ({100.0*i/nBatches:<5.4}%)]"
                f"    Loss: {trainLoss/verbosity:>10.8f}    Accuracy: {trainAcc/verbosity:>10.8f}"
            )
            trainLoss = 0.0
            trainAcc = 0.0

    msg = (f"    Avg train loss: {avgTrainLoss/nBatches:>15.4f} \n"
           f"    Avg train accuracy: {avgTrainAcc/nBatches:>11.4f}\n")

    if validationLoader is not None:
        validAcc = model.accuracy(validationLoader)
        msg += f"    Validation accuracy: {validAcc:>10.4f}"

    print(msg+"\n")


def test(dataLoader, model):
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
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
               'ship', 'truck')
    totCorrect = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data, labels in dataLoader:
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct = (predicted == labels)
            totCorrect += correct.sum().item()
            c = correct.squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    print('Test accuracy: %d %%' % (
        100 * totCorrect / total))


def learn(trainDataLoader, validationDataLoader, testDataLoader, model, epochs, 
          learningRate=1e-3, momentum=0.9, verbosity=5):
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
        Learning rate of the SGD optimizer. Default 0.001.
    momentum : `float`, optional
        Momentum of the SGD optimizer, default: 0.9
    verbosity: `int`, optional
        How often are batch losses printed, larger number means less prints.
        Epoch average loss is always printed. Default: 20
    """
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    for epoch in range(1, epochs + 1):
        train(trainDataLoader, model, optimizer, epoch, verbosity=verbosity, 
              validationLoader=validationDataLoader)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class ConvolutionalNeuralNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lossFunction = kwargs.pop("lossFunction", nn.CrossEntropyLoss())

    def forward(self, x):
        raise NotImplementedError("Forward must be implemented by child class!")

    def loss(self, data, labels=None):
        """Given the data, alculate MSE loss of the reconstruction. Ensure the
        autoencoder has been trained.

        Parameters
        -----------
        x : `torch.tensor`
            Data 

        Returns
        -------
        loss : `nn.CrossEntropy`
            Cross entropy Loss
        """
        reconstructed = self.forward(data)
        labels = data.view(-1, self.imgSize) if labels is None else labels
        loss = self.lossFunction(reconstructed, labels)
        return loss

    def nFeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def _tensorAccuracy(self, data, labels):
        with torch.no_grad():
            outputs = self.forward(data)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct, total

    def accuracy(self, data, labels=None):
        total, correct = 0, 0
        if isinstance(data, datutils.DataLoader):
            for data, labels in data:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                good, tot = self._tensorAccuracy(data, labels)
                total += tot
                correct += good
        elif labels is not None:
            correct, total = self._tensorAccuracy(data, labels)
        return 100 * correct/total


class TutorialNet(ConvolutionalNeuralNet):
    def __init__(self, M1=6, M2=16, k1=120, k2=84, *args, **kwargs):
        super().__init__(**kwargs)
        self.forward1 = nn.Sequential(
            nn.Conv2d(3, M1, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(M1, M2, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.forward2 = nn.Sequential(
            nn.Linear(M2*5*5, k1),
            nn.ReLU(),
            nn.Linear(k1, k2),
            nn.ReLU(),
            nn.Linear(k2, 10)
        )
        self.M1 = M1
        self.M2 = M2
        self.k1 = k1
        self.k2 = k2

    def forward(self, x):
        y = self.forward1(x)
        z = y.view(-1, self.nFeatures(y))
        return self.forward2(z)


class A5a(ConvolutionalNeuralNet):
    def __init__(self, *args, **kwargs):
        self.linear = nn.Linear(3072, 10)

    def forward(self, x):
        x = x.view(-1, self.nFeatures(x))
        return self.linear(x)


class A5b(ConvolutionalNeuralNet):
    def __init__(self, N, M, k, *args, **kwargs):
        finSize = int( ((33 - k) / N)**2 * M)
        self.forward1 = nn.Sequential(
            nn.Linear(3072, M),
            nn.ReLU(),
            nn.Linear(M, 10)
        )

    def forward(self, x):
        y = x.view(-1, self.nFeatures(x))
        return self.forward1(x)


class A5c(ConvolutionalNeuralNet):
    def __init__(self, N, M, k, *args, **kwargs):
        finSize = int( ((33 - k) / N)**2 * M)
        self.forward1 = nn.Sequential(
            nn.Conv2d(3, M, k),
            nn.ReLU(),
            nn.MaxPool2d(N, N)
        )
        self.forward2 = nn.Conv2d(finSize, 10)
        self.N = N
        self.M = M
        self.k = k
        self.finSize = finSize

    def forward(self, x):
        y = self.forward1(x)
        z = y.view(-1, self.nFeatures(y))
        return self.forward2(x)





PATH = 'data/cifarTrained.pth'
trainData, validationData, testData = load_cifar_dataset(batchSize=4)


net = TutorialNet()
#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#    net = nn.DataParallel(net)
net = net.to(DEVICE)
learn(trainData, validationData, testData, net, 2)
torch.save(net.state_dict(), PATH)







test(testData, net)
