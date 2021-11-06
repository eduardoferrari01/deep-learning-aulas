import torch
from torch import nn
from sklearn import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

diabetes = datasets.load_diabetes()

data = diabetes.data
target = diabetes.target

print(data.shape, target.shape)
print(diabetes.feature_names)


# Implementando o MLP
class DiabetesRegression(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super(DiabetesRegression, self).__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, out_size)
        self.softmax = nn.Softmax()

    def forward(self, X):
        feature = self.relu(self.hidden(X))
        output = self.softmax(self.out(feature))
        return output


input_size = data.shape[1]
hidden_size = 32
out_size = 1

net = DiabetesRegression(input_size, hidden_size, out_size).to(device)

criterion = nn.MSELoss().to(device)

Xtns = torch.from_numpy(data).float().to(device)
Ytns = torch.from_numpy(target).float().to(device)

print(Xtns.shape, Ytns.shape)

pred = net(Xtns)
print(pred.shape)

loss = criterion(pred.squeeze(), Ytns)
print(loss.data)
