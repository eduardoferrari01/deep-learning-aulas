import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

np.random.seed(20)

X, Y = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)


def plotmodel(w1, w2, b):
    # plot da distribuição
    # scatter = espalhar dados. c=Y vetor de classes
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, edgecolors='k')

    xmim, xmax = plt.gca().get_xlim()
    ymim, ymax = plt.gca().get_ylim()

    x = np.linspace(-2, 4, 50)
    y = (-w1 * x - b) / w2  # antigo y = (-a * x - c) / b

    plt.axvline(0, -1, 1, color='k', linewidth=1)
    plt.axhline(0, -2, 4, color='k', linewidth=1)
    plt.plot(x, y)
    plt.grid(True)
    plt.xlim(xmim, xmax)
    plt.ylim(ymim, ymax)


p1 = X[10];
print(Y[10])

w1 = 1.3  # a
w2 = 10  # b
b = -0.1  # c

plotmodel(w1, w2, b)


def classify(ponto, w1, w2, b):
    ret = w1 * ponto[0] + w2 * ponto[1] + b

    if ret >= 0:
        return 1, 'yellow'
    else:
        return 0, 'blue'


p = (2, -1)
# print(w1*p[0] + w2 * p[1] + b)

classe, cor = classify(p, w1, w2, b)
print(classe, cor)

plotmodel(w1, w2, b)

# plt.plot(p[0], p[1], marker='^', markersize=20)

acertos = 0;

for k in range(len(X)):
    categ, _ = classify(X[k], w1, w2, b)
    if categ == Y[k]:
        acertos += 1

print("Acurácia =  {0}".format(100 * acertos / len(X)))

plt.show()
