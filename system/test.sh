#!/bin/bash

# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedAvg -gr 1000 -did 5 -go cnn -fr 1500 > outputs/cifar10_fedavg.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 7 -go cnn -fr 1500 > outputs/cifar10_fedmgda_20.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 7 -go cnn -fr 1500 > outputs/cifar10_fedmgda_20.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 6 -go cnn -fr 1500 -bt 0.5 > outputs/cifar10_fedmoco_20.out 2>&1 &

# fedmgda
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 7 -go cnn -fr 1500 > outputs/cifar10_fedmgda_10.out 2>&1 &
nohup python -u main.py -nc 5 -data Cifar10 -m cnn -algo FedMoCo -gr 4 -did 2 -go cnn -fr 1500 -bt 0.1 > outputs/test_cifar10_fedmgda_01.out 2>&1 &
nohup python -u main.py -nc 5 -data Cifar10 -m cnn -algo FedMoCo -gr 4 -did 2 -go cnn -fr 1500 -bt 0.3 > outputs/test_cifar10_fedmgda_03.out 2>&1 &
nohup python -u main.py -nc 5 -data Cifar10 -m cnn -algo FedMoCo -gr 4 -did 2 -go cnn -fr 1500 -bt 0.5 > outputs/test_cifar10_fedmgda_05.out 2>&1 &
nohup python -u main.py -nc 5 -data Cifar10 -m cnn -algo FedMoCo -gr 4 -did 4 -go cnn -fr 1500 -bt 0.7 > outputs/test_cifar10_fedmgda_07.out 2>&1 &
nohup python -u main.py -nc 5 -data Cifar10 -m cnn -algo FedMoCo -gr 4 -did 5 -go cnn -fr 1500 -bt 0.9 > outputs/test_cifar10_fedmgda_09.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 7 -go cnn -fr 1500 > outputs/cifar10_fedmgda_20.out 2>&1 &

# fedavg
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 5 -go cnn -fr 1 -bt 1 > outputs/cifar10_fedavg_10.out 2>&1 &
# nohup python -u main.py -nc 5 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 5 -go cnn -fr 1 -bt 1 > outputs/cifar10_fedavg_5.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 5 -go cnn -fr 1 -bt 1 > outputs/cifar10_fedavg_20.out 2>&1 &

# fedmoco
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 6 -go cnn -fr 1500 -bt 0.5 > outputs/cifar10_fedmoco_10.out 2>&1 &
# nohup python -u main.py -nc 5 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 6 -go cnn -fr 1500 -bt 0.5 > outputs/cifar10_fedmoco_5.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 6 -go cnn -fr 1500 -bt 0.5 > outputs/cifar10_fedmoco_20.out 2>&1 &

# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMoCo -gr 10 -did 6 -go cnn -fr 1500 -bt 0.5 > outputs/cifar10_fedmoco2.out 2>&1 &
# python main.py -nc 10 -data Cifar10 -m cnn -algo FedAvg -gr 1000 -did 6 -go cnn


# python main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1500 -did 6 -go cnn -fr 1000 -bt 0.5

