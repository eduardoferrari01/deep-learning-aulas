import matplotlib
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time, os

# Configurando hiperparâmetros.
args = {
    'epoch_num': 2,  # Número de épocas.
    'lr': 1e-3,  # Taxa de aprendizado.
    'weight_decay': 1e-3,  # Penalidade L2 (Regularização).
    'batch_size': 50,  # Tamanho do batch.
}

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

print(args['device'])

data_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(), ])

train_set = datasets.CIFAR10('.',
                             train=True,
                             transform=data_transform,
                             download=True)

test_set = datasets.CIFAR10('.',
                            train=False,
                            transform=data_transform,
                            download=False)

train_loader = DataLoader(train_set,
                          batch_size=args['batch_size'],
                          shuffle=True)

test_loader = DataLoader(test_set,
                         batch_size=args['batch_size'],
                         shuffle=True)

net = nn.Sequential(
    ## ConvBlock 1
    nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),  # entrada: (b, 3, 32, 32) e saida: (b, 6, 28, 28)
    nn.BatchNorm2d(6),
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),  # entrada: (b, 6, 28, 28) e saida: (b, 6, 14, 14)

    ## ConvBlock 2
    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # entrada: (b, 6, 14, 14) e saida: (b, 16, 10, 10)
    nn.BatchNorm2d(16),
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),  # entrada: (b, 16, 10, 10) e saida: (b, 16, 5, 5)

    ## ConvBlock 3
    nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),  # entrada: (b, 16, 5, 5) e saida: (b, 120, 1, 1)
    nn.BatchNorm2d(120),
    nn.Tanh(),
    nn.Flatten(),
    # lineariza formando um vetor                # entrada: (b, 120, 1, 1) e saida: (b, 120*1*1) = (b, 120)

    ## DenseBlock
    nn.Linear(120, 84),  # entrada: (b, 120) e saida: (b, 84)
    nn.Tanh(),
    nn.Linear(84, 10),  # entrada: (b, 84) e saida: (b, 10)
)

net = net.to(args['device'])

criterion = nn.CrossEntropyLoss().to(args['device'])
optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])


def train(train_loader, net, epoch):
    net.train()

    start = time.time()

    epoch_loss = []
    pred_list, rotulo_list = [], []
    for batch in train_loader:
        dado, rotulo = batch

        dado = dado.to(args['device'])
        rotulo = rotulo.to(args['device'])

        ypred = net(dado)
        loss = criterion(ypred, rotulo)
        epoch_loss.append(loss.cpu().data)

        _, pred = torch.max(ypred, axis=1)
        pred_list.append(pred.cpu().numpy())
        rotulo_list.append(rotulo.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = np.asarray(epoch_loss)
    pred_list = np.asarray(pred_list).ravel()
    rotulo_list = np.asarray(rotulo_list).ravel()

    acc = accuracy_score(pred_list, rotulo_list)

    end = time.time()
    print('#################### Train ####################')
    print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f' % (
        epoch, epoch_loss.mean(), epoch_loss.std(), acc * 100, end - start))

    return epoch_loss.mean()


def validate(test_loader, net, epoch):

    net.eval()

    start = time.time()

    epoch_loss = []
    pred_list, rotulo_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            dado, rotulo = batch

            dado = dado.to(args['device'])
            rotulo = rotulo.to(args['device'])

            ypred = net(dado)
            loss = criterion(ypred, rotulo)
            epoch_loss.append(loss.cpu().data)

            _, pred = torch.max(ypred, axis=1)
            pred_list.append(pred.cpu().numpy())
            rotulo_list.append(rotulo.cpu().numpy())

    epoch_loss = np.asarray(epoch_loss)
    pred_list = np.asarray(pred_list).ravel()
    rotulo_list = np.asarray(rotulo_list).ravel()

    acc = accuracy_score(pred_list, rotulo_list)

    end = time.time()
    print('********** Validate **********')
    print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f\n' % (
        epoch, epoch_loss.mean(), epoch_loss.std(), acc * 100, end - start))

    return epoch_loss.mean()


train_losses, test_losses = [], []
for epoch in range(args['epoch_num']):

    train_losses.append(train(train_loader, net, epoch))

    test_losses.append(validate(test_loader, net, epoch))
