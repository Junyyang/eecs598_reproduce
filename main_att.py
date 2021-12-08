import torch
import numpy as np
import os
import argparse
import time

random_seed = 6667
# 6666 for horse
#6667 for nothing
np.random.seed(random_seed)
torch.manual_seed(random_seed)

from model.Server import Server
from model.Client import Client
from model.Adv_Client import AdvClient
from model.Server_attack import Server_att
from model.Vic_Client import VicClient
from utils import load_mnist, sample_iid, load_cifar, load_LFW
from conf import Args

if __name__ == "__main__":
    time_begin = time.time()
    args = Args()

    '''
    
    parser = argparse.ArgumentParser(description='Sketching')
    parser.add_argument('--model_type', type=str, default="CNN")
    parser.add_argument('--datatype', type=str, default="cifar")
    parser.add_argument('--sketchtype', type=str, default="count")
    parser.add_argument('--round', type=int, default=500)
    parser.add_argument('--savingmodel', type=int, default=0)

    args = parser.parse_args()
    for ki in old_args.keys():
        if not (ki in args.keys()):
            args[ki] = old_args[ki]
    '''

    if not os.path.exists('data/results'):
        os.makedirs('data/results')

    if args.datatype == 'mnist':
        path = './data/mnist'
        train_data, test_data = load_mnist(path)
    elif args.datatype == 'cifar':
        path = './data/cifar'
        train_data, test_data = load_cifar(path)
    # # for Labeled Faces In the Wild
    elif args.datatype == 'LFW':
        path = './data/LFW'
        train_data, test_data = load_LFW(path)


    data_split = sample_iid(train_data, args.number_client)


    print("model type: ", args.model_type)
    print("dataset: ", args.datatype)
    print("target test accuracy: ", args.target)
    print("sketch method: ", args.sketchtype)
    print("attack status: ", args.attack)
    print("Start Training")
    print("======================")
    print("")
    print("")


    # ====================================================
    # adversory client added
    num_vic_adv = 2
    data_split_2 = sample_iid(train_data, num_vic_adv)
    vic_client = VicClient(train_data, data_split_2[0], args)
    adv_client = AdvClient(train_data, data_split_2[1], args)

    attacked_server = Server_att( [adv_client, vic_client], test_data, args, attack = True)
    attacked_server.init_paras()
    attacked_server.train()
    time_end = time.time()

    print('The attacked training takes {} seconds.'.format(time_end-time_begin))

    # # # ====================================================
    # # # Not attacked server training
    # clients = []
    # for i in range(args.number_client):
    #     client = Client(train_data, data_split[i], args)
    #     clients.append(client)
    # server = Server(clients, test_data, args, attack = False)

    # server.init_paras()
    # # from torchsummary import summary
    # # summary(server.server_model, (1, 784))
    # # exit()
    # server.train()
    # time_end = time.time()
    # print('The entire training takes {} seconds.'.format(time_end-time_begin))


