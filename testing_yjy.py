import torch
import numpy as np
import os

random_seed = 6666
np.random.seed(random_seed)
torch.manual_seed(random_seed)

from model.Server import Server
from model.Client import Client

from utils import load_mnist, sample_iid, load_cifar, load_LFW
from conf import Args


args = Args()

if not os.path.exists('data/results'):
    os.makedirs('data/results')

if args.datatype == 'mnist':
    path = './data/mnist'
    train_data, test_data = load_mnist(path)
elif args.datatype == 'cifar':
    path = './data/cifar'
    train_data, test_data = load_cifar(path)
# for Labeled Faces In the Wild
elif args.datatype == 'LFW':
    path = './data/LFW'
    train_data, test_data = load_LFW(path)

print("type of train_data", type(train_data))