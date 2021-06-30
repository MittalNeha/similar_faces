import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .dataset import FaceAttributesDataset
from .model import MCNN

# variables define
baseFolder = '/content/'
imgFolder = 'img_align_celeba/img_align_celeba'
use_cuda = torch.cuda.is_available()
batch_size = 16
device = torch.device("cuda" if use_cuda else "cpu")

train_transform = transforms.Compose([
    # other transformations to be added in this list
    transforms.ToTensor()
])

kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

train_dataset = FaceAttributesDataset(baseFolder, imgFolder, 'list_attr_celeba.csv', transform=train_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True, **kwargs)

print("batch_size is {}", batch_size)
# TODO: test_dataset

# MCNN model to be trained
net = MCNN().to(device)


# summary(net, input_size=(3, 227, 227))

def get_acc(out, target):
    out[out >= 0.5] = 1
    out[out < 0.5] = 0
    acc = (out == target).sum() / (batch_size * target.shape[1]) * 100

    return acc


def calculate_criterion(criterion, labels, output):
    loss = 0
    total_acc = 0
    for key in output.keys():
        # print(key)
        out = output[key].cpu()
        target = labels[key].float()
        loss += criterion(out, target)
        total_acc += get_acc(out, target)
    total_loss = torch.tensor(loss, requires_grad=True)  # sourceTensor.clone().detach().requires_grad_(True)
    total_acc /= len(output.keys())
    return total_loss, total_acc


# make groups for the labels as per the Network
def train(criterion, optimizer, schedular=None):
    net.train()

    train_loss = 0
    train_accuracy = 0
    num_loops = 0

    for batch_idx, sample in enumerate(train_loader):
        data, target = sample['image'].to(device), sample['attr']
        # data, target = sample['image'].to(device), sample['attr'].to(device)
        # print(data)
        optimizer.zero_grad()

        output = net(data)

        # calculate the loss and accuracy
        loss, acc = calculate_criterion(criterion, target, output)

        loss.backward()
        optimizer.step()
        # schedular.step()

        train_loss += loss.item()
        train_accuracy += acc
        num_loops += 1
        if batch_idx % 100 == 0:
            print("{}: loss {}, acc {}".format(num_loops, loss, acc))
    train_accuracy = train_accuracy / num_loops

    return train_accuracy, train_loss


def fit_model(epochs=5):
    train_acc = []
    test_acc = []

    train_loss = []
    test_loss = []

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.9, )

    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=2, verbose=True)

    for epoch in range(1, epochs):
        acc, loss = train(criterion, optimizer, scheduler)
        train_loss.append(loss)
        train_acc.append(acc)

        # TODO: call test function and calulate test acc and loss

    return train_loss, train_acc, test_loss, test_acc


def attributes_model():
    # Train the MCNN model
    hist = fit_model(2)
    return net, hist


def main():
    # Train the MCNN model
    hist = fit_model(2)


if __name__ == '__main__':
    main()
