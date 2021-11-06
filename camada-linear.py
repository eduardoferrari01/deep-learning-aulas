import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(42)

perceptron = nn.Linear(3, 1)


# for name, tensor in perceptron.named_parameters():
#    print(name, tensor.data)

# ou acesso direto
# print('#######')
# print(perceptron.weight.data)
# print(perceptron.bias.data)

def plot3d(perceptron):
    w1, w2, w3 = perceptron.weight.data.numpy()[0]
    b = perceptron.bias.data.numpy()
    # w1 * x1 + w2 * x2 + w3 * x3 + b = 0

    X1 = np.linspace(-1, 1, 10)  # fixando  dimensão 1
    X2 = np.linspace(-1, 1, 10)  # fixando  dimensão 2

    X1, X2 = np.meshgrid(X1, X2)  # criando o mesh 2D

    X3 = (b - w1 * X1 - w2 * X2) / w3  # dimensão 3

    # 3d
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')
    ax.view_init(azim=180)
    ax.plot_surface(X1, X2, X3, cmap='plasma')


X = torch.Tensor([0, 1, 2])

y = perceptron(X)

print(y)

plot3d(perceptron)

plt.plot([X[0]], [X[1]], [X[2]], color='r', marker='^', markersize=20)

plt.show()
