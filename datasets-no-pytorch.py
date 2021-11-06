import torch
from torch import nn, optim
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

args = {
    'batch_size': 20,
    'num_workers': 4
}

if torch.cuda.is_available():
    args['device'] = torch.device('gpu')
else:
    args['device'] = torch.device('cuda')

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

print('Amostras de treino:' + str(len(train_set)) + '\Amostra de teste' + str(len(test_set)))

print(type(train_set))
print(type(train_set[0]))
# print(train_set[0])


# for i in range(3):
#     dado, rotulo = train_set[i]
#
#     plt.figure()
#     plt.imshow(dado[0])
#     plt.title('Rotulo:' + str(rotulo))
#     #plt.show()

train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])

test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])

for batch in train_loader:
    dado, rotulo = batch
    print(dado.size(), rotulo.size())
    break
