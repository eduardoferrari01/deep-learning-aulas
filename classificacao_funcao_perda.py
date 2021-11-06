import torch
from torch import nn
from sklearn import datasets;

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

wine = datasets.load_wine()

data = wine.data
target = wine.target

print(data.shape, target.shape)
print(wine.feature_names, wine.target_names)


class WineClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super(WineClassifier, self).__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, out_size)
        self.softmax = nn.Softmax()  # faz distribuição de probabilidade

    def forward(self, X):
        feature = self.relu(self.hidden(X))
        output = self.softmax(self.out(feature))
        return output


input_size = data.shape[1]
hidden_size = 32  # hiperparâmetros
out_size = len(wine.target_names)

net = WineClassifier(input_size, hidden_size, out_size).to(device)

print(net)

criterion = nn.CrossEntropyLoss().to(device)

Xtns = torch.from_numpy(data).float()
Ytns = torch.from_numpy(target)

Xtns = Xtns.to(device)
Ytns = Ytns.to(device)

print(Xtns.dtype, Ytns.dtype)

pred = net(Xtns)

print(Xtns.shape)
print(pred.shape)

print(pred.shape, Ytns.shape)

loss = criterion(pred, Ytns)  # calcula a perda media
print(loss)
