from builtins import int

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X1, Y1 = make_moons(n_samples=300, noise=0.2)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')

input_size = 2
hidden_size = 8
output_size = 1

net = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size),  # camada hidden(escondida)
                    nn.ReLU(),  # camada ativação não linear
                    nn.Linear(in_features=hidden_size, out_features=output_size))  # camada output (saída)


# print(net)
# summary(net, input_size=(1, input_size))

# forward
# print(X1.dtype)
# tensor = torch.from_numpy(X1).float()
# pred = net(tensor)
# print(pred.size())

class MinhaRede(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MinhaRede, self).__init__()
        # Definir a arquitetura
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # ativacao não linear
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # Gera uma saída a partir do X
        hidden = self.relu(self.hidden(X))
        output = self.output(hidden)

        return output


net2 = MinhaRede(input_size, hidden_size, output_size)

print(X1.dtype)
tensor = torch.from_numpy(X1).float()
pred = net2(tensor)
print(pred.size())

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)
net2 = MinhaRede(input_size, hidden_size, output_size)
net2 = net2.to(device)
print(net)

print((X1.shape))
tensor = torch.from_numpy(X1).float()
tensor = tensor.to(device)
pred = net2(tensor)
print(pred.size())
