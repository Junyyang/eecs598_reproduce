"""
Load data:

load and preprocess data

sample:
sample the data in an IID way.

"""

# reference code:
# https://github.com/tamerthamoqa/facenet-pytorch-glint360k/blob/b1d8b1014b00650688646330fcd258728c7ccb2f/misc/test_model_lfw_far.py
# https://github.com/liorshk/facenet_pytorch/blob/master/LFWDataset.py

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from testing_LFWDataset import LFWDataset
import numpy as np
from split_dataset import split
from conf import Args


args = Args()


def load_mnist(path):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST(path, train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST(path, train=False, download=True, transform=trans_mnist)
    return dataset_train, dataset_test


def load_cifar(path_cifar):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.8, saturation=0.05, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset_train = datasets.CIFAR10(path_cifar, train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10(path_cifar, train=False, download=True, transform=transform_test)
    return dataset_train, dataset_test

# ================
# Labeled face in the wild
def load_LFW(lfw_dir):

    lfw_transforms = transforms.Compose([
                        transforms.Resize(size=[int(32), int(32)]), 
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.6071, 0.4609, 0.3944],
                            std=[0.2457, 0.2175, 0.2129]
                        )
                        ])

    # lfw_dir = './lfw/lfw_gender'
    lfw_dataset = ImageFolder(lfw_dir, transform = lfw_transforms)
    n = int(len(lfw_dataset)*0.8)
    dataset_train, dataset_test = split(lfw_dataset, n, seed=0)

    return dataset_train, dataset_test


# Randomly shuffle the training data and dispense training data to different users
# Generate random indices and random signs
# Args:
#    dataset: training data
#    num_users: number of users
# Return:
#    data_split: a list of indices for users
def sample_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    data_split = []
    idx = np.random.permutation(len(dataset))
    for i in range(num_users):
        start = i * num_items
        end = start + num_items
        data_split.append(idx[start:end])
    return data_split


if __name__ == '__main__':
    train_data, test_data = load_mnist('./data/mnist')
