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

path = './data/lfw/lfw_gender'
train_data, test_data = load_LFW(path)

data_split = sample_iid(train_data, args.number_client)
print('data_split', data_split)

print("model type: ", args.model_type)
print("dataset: ", args.datatype)
print("target test accuracy: ", args.target)

clients = []
for i in range(args.number_client):
    client = Client(train_data, data_split[i], args)
    clients.append(client)

server = Server(clients, test_data, args)

server.init_paras()
# # from torchsummary import summary
# # summary(server.server_model, (1, 784))
# # exit()
server.train()


