import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "Cifar10/"


# Allocate data to users
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    trainset_image = trainset.data.cpu().detach().numpy()
    testset_image = testset.data.cpu().detach().numpy()

    trainset_label = trainset.targets.cpu().detach().numpy()
    testset_label = testset.targets.cpu().detach().numpy()

    train_X, train_y, train_statistic = separate_data((trainset_image, trainset_label), num_client, num_classes, niid, balance, partition)
    test_X, test_y, test_statistic = separate_data((testset_image, testset_label), num_client, num_classes, niid, balance, partition)

    train_data = []
    test_data = []

    train_samples = []
    test_samples = []
    
    for i in num_client:
        train_data.append({'x': train_X[i]}, 'y': train_y[i])
        test_data.append({'x': test_X[i]}, 'y': test_y[i])
        
        train_samples.append(len(train_y[i]))
        test_samples.append(len(test_y[i]))

    statistic = train_statistic

    print("Total number of samples:", sum(train_samples + test_samples))
    print("The number of train samples:", train_samples)
    print("The number of test samples:", test_samples)
    print()
    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition)
