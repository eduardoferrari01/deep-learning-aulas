from _ast import arg

import torch
from torch import nn, optim
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

args = {
    'batch_size': 20,
    'num_workers': 4,
    'num_classes': 10,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'num_epochs': 30
}

# args = {
#     'batch_size': 500,
#     'num_workers': 4,
#     'num_classes': 10,
#     'lr': 1e-4,
#     'weight_decay': 5e-4,
#     'num_epochs': 30
# }


if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

print(args['device'])

# Carrega o datasets para treino
train_set = datasets.MNIST('/dataset/',
                           train=True,
                           transform=transforms.ToTensor(),
                           # transform=transforms.RandomCrop(12),
                           download=True)

# Carrega o datasets para teste
test_set = datasets.MNIST('/dataset/',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False)

train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])

test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super(MLP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_size, out_size)
        self.softmax = nn.Softmax()

    def forward(self, X):
        X = X.view(X.size(0), -1)

        feature = self.features(X)
        output = self.softmax(self.out(feature))

        return output


input_size = 28 * 28
hidden_size = 128
out_size = 10

net = MLP(input_size, hidden_size, out_size).to(args['device'])

criterion = nn.CrossEntropyLoss().to(args['device'])
optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

for epoch in range(args['num_epochs']):

    epoch_loss = []
    for batch in train_loader:
        dado, rotulo = batch

        dado = dado.to(args['device'])
        rotulo = rotulo.to(args['device'])

        # Forward
        pred = net(dado)
        loss = criterion(pred, rotulo)
        epoch_loss.append(loss.cpu().data)

        # Backwar
        loss.backward()
        optimizer.step()

    epoch_loss = np.asarray(epoch_loss)

    print("Epoca %d, Loss: %.4f +\- %.4f" % (epoch, epoch_loss.mean(), epoch_loss.std()))
