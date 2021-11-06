import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch import optim

features = [0, 9]

wine = datasets.load_wine()

data = wine.data[:, features]
scaler = StandardScaler()
data = scaler.fit_transform(data)
targets = wine.target

# plt.scatter(data[:, 0], data[:, 1], c=targets, s=15, cmap=plt.cm.brg)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

input_size = data.shape[1]
hidden_size = 32
out_size = len(wine.target_names)

net = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, out_size),
    nn.Softmax()
)

net.to(device)


def plot_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    spacing = min(x_max - x_min, y_max - y_min) / 100

    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))

    data = np.hstack((XX.ravel().reshape(-1, 1),
                      YY.ravel().reshape(-1, 1)))

    # For binary problems
    # db_prob = model(Variable(torch.Tensor(data)).cuda())
    # clf = np.where(db_prob.cpu().data < 0.5, 0, 1)

    # For multi-class problems
    db_prob = model(torch.Tensor(data).to(device))
    clf = np.argmax(db_prob.cpu().data.numpy(), axis=-1)

    Z = clf.reshape(XX.shape)

    plt.contourf(XX, YY, Z, cmap=plt.cm.brg, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=25, cmap=plt.cm.brg)


criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net.parameters(), lr=1e-3)

plt.xlabel(wine.feature_names[features[0]])
plt.ylabel(wine.feature_names[features[1]])

X = torch.FloatTensor(data).to(device)
Y = torch.LongTensor(targets).to(device)

# plt.ion()
# fig = plt.figure()

for i in range(300):
    # Forward
    pred = net(X)
    loss = criterion(pred, Y)  # Calcula a função de custo

    # Backpropagation
    loss.backward()  # Calcula o gradiente
    optimizer.step()  # Atualizar os pesos

    if i % 10 == 0:
        plot_boundary(data, targets, net)
        # fig.canvas.draw()
        # fig.canvas.flush_events()

print("FIM do treinamento")
plt.show()
