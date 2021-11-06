import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

args = {
    'batch_size': 20,
    'num_workers': 4,
    'num_classes': 10,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'num_epochs': 200
}

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

df = pd.read_csv('/dataset/Bike-Sharing-Dataset/hour.csv')

torch.manual_seed(1)
indices = torch.randperm(len(df)).tolist()

train_size = int(0.8 * len(df))  # pega 80 dos 17379 %
df_train = df.iloc[indices[:train_size]]
df_test = df.iloc[indices[train_size]]

df_train.to_csv('/dataset/Bike-Sharing-Dataset/bike_train.csv',
                index=False)
df_test.to_csv('/dataset/Bike-Sharing-Dataset/bike_test.csv',
               index=False)


class Bicicletinha(Dataset):
    def __init__(self, csv_path):
        self.dados = pd.read_csv(csv_path).to_numpy()

    def __getitem__(self,
                    idx):
        sample = self.dados[idx][2:14]
        label = self.dados[idx][-1:]

        # converter pra tensor
        sample = torch.from_numpy(sample.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        return sample, label

    def __len__(self):
        return len(self.dados)


train_set = Bicicletinha(
    '/dataset/Bike-Sharing-Dataset/bike_train.csv')
test_set = Bicicletinha('/dataset/Bike-Sharing-Dataset/bike_test.csv')

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

    def forward(self, X):
        feature = self.features(X)
        output = self.out(feature)

        return output


input_size = len(train_set[0][0])
hidden_size = 128
out_size = 1

net = MLP(input_size, hidden_size, out_size).to(args['device'])

criterion = nn.L1Loss().to(args['device'])
optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])


def train(train_loader, net, epoch):
    net.train()
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
    return epoch_loss.mean()


def test(test_loader, net, epoch):
    net.eval()
    with torch.no_grad():
        epoch_loss = []
        for batch in train_loader:
            dado, rotulo = batch

            dado = dado.to(args['device'])
            rotulo = rotulo.to(args['device'])

            # Forward
            pred = net(dado)
            loss = criterion(pred, rotulo)
            epoch_loss.append(loss.cpu().data)

        epoch_loss = np.asarray(epoch_loss)
        print("Epoca %d, Loss: %.4f +\- %.4f" % (epoch, epoch_loss.mean(), epoch_loss.std()))
        return epoch_loss.mean()


train_losses, test_losses = [], []
for epoch in range(args['num_epochs']):
    # Train
    train_losses.append(train(train_loader, net, epoch))

    # Validate
    test_losses.append(test(test_loader, net, epoch))
    print('--------------------------')

plt.figure(figsize=(20, 9))
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test', linewidth=3, alpha=0.5)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Convergence', fontsize=16)
plt.legend()
plt.show()
