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
    'num_epochs': 30
}

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

df = pd.read_csv('/dataset/Bike-Sharing-Dataset/hour.csv')
# print(len(df))
# df.head()
# print(df.head()) # ou print(df)

torch.manual_seed(1)
indices = torch.randperm(len(df)).tolist()

train_size = int(0.8 * len(df))
df_train = df.iloc[indices[:train_size]]
df_test = df.iloc[indices[train_size]]

# print(len(df_train), len(df_test))

df_train.to_csv('/dataset/Bike-Sharing-Dataset/bike_train.csv', index=False)
df_test.to_csv('/dataset/Bike-Sharing-Dataset/bike_test.csv', index=False)


class Bicicletinha(Dataset):
    def __init__(self, csv_path):
        self.dados = pd.read_csv(csv_path).to_numpy()

    def __getitem__(self, idx):
        sample = self.dados[idx][2:14]
        label = self.dados[idx][-1:]

        # converter pra tensor
        sample = torch.from_numpy(sample.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        return sample, label

    def __len__(self):
        return len(self.dados)


train_set = Bicicletinha('/dataset/Bike-Sharing-Dataset/bike_train.csv')
test_set = Bicicletinha('/dataset/Bike-Sharing-Dataset/bike_test.csv')

# dados, rotulo = train_set[0]
# print(rotulo)
# print(dados)

train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])

for batch in train_loader:
    dado, rotulo = batch
    print(dado.size(), rotulo.size())
    break
