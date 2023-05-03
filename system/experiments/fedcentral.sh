#!/bin/bash

nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedCentral -lbs 16 -gr 1000 -rh 1  -did 3 -go cnn -fr 0 -tid 10131 -cdn 1 -cdd Cifar10_central > outputs/cifar10_fedavg_10_10131.out 2>&1 &
nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedCentral -lbs 16 -gr 1000 -rh 1  -did 4 -go cnn -fr 0 -tid 10141 -cdn 10 -cdd Cifar10_iid > outputs/cifar10_fedavg_10_10141.out 2>&1 &
nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedCentral -lbs 16 -gr 1000 -rh 1  -did 5 -go cnn -fr 0 -tid 10151 -cdn 10 -cdd Cifar10 > outputs/cifar10_fedavg_10_10151.out 2>&1 &

# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedCentral -lbs 16 -gr 1000 -rh 1  -did 3 -go cnn -fr 0 -tid 10132 -cdn 1 -cdd Cifar10_central > outputs/cifar10_fedavg_10_10132.out 2>&1 &
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedCentral -lbs 16 -gr 1000 -rh 1  -did 4 -go cnn -fr 0 -tid 10142 -cdn 10 -cdd Cifar10_iid > outputs/cifar10_fedavg_10_10142.out 2>&1 &
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedCentral -lbs 16 -gr 1000 -rh 1  -did 5 -go cnn -fr 0 -tid 10152 -cdn 10 -cdd Cifar10 > outputs/cifar10_fedavg_10_10152.out 2>&1 &

# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedCentral -lbs 64 -gr 1000 -rh 1  -did 3 -go cnn -fr 0 -tid 10133 -cdn 1 -cdd Cifar10_central > outputs/cifar10_fedavg_10_10133.out 2>&1 &
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedCentral -lbs 64 -gr 1000 -rh 1  -did 4 -go cnn -fr 0 -tid 10143 -cdn 10 -cdd Cifar10_iid > outputs/cifar10_fedavg_10_10143.out 2>&1 &
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedCentral -lbs 64 -gr 1000 -rh 1  -did 5 -go cnn -fr 0 -tid 10153 -cdn 10 -cdd Cifar10 > outputs/cifar10_fedavg_10_10153.out 2>&1 &



# python main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1500 -did 6 -go cnn -fr 1000 -bt 0.5

