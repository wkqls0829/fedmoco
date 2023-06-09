#!/bin/bash
# join ratio experiment 
nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -jr 1.0 -did 0 -go cnn -fr 0 -bt 1 -tid 10001 > outputs/cifar10_fedavg_20_10001.out 2>&1 &
nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -jr 0.5 -did 1 -go cnn -fr 0 -bt 1 -tid 10002 > outputs/cifar10_fedavg_20_10002.out 2>&1 &
nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -jr 0.25 -did 2 -go cnn -fr 0 -bt 1 -tid 10003 > outputs/cifar10_fedavg_20_10003.out 2>&1 &

nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -jr 1.0 -did 3 -go cnn -fr 2000 -rh 1 -tid 20001 > outputs/cifar10_fedmgda_20_20001.out 2>&1 &
nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -jr 0.5 -did 4 -go cnn -fr 2000 -rh 1 -tid 20002 > outputs/cifar10_fedmgda_20_20002.out 2>&1 &
nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -jr 0.25 -did 5 -go cnn -fr 2000 -rh 1 -tid 20003 > outputs/cifar10_fedmgda_20_20003.out 2>&1 &

nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -jr 1.0 -did 6 -go cnn -fr 2000 -rh 1 -bt 0.5 -tid 30001 > outputs/cifar10_fedmoco_20_30001.out 2>&1 &
nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -jr 0.5 -did 7 -go cnn -fr 2000 -rh 1 -bt 0.5 -tid 30002 > outputs/cifar10_fedmoco_20_30002.out 2>&1 &
nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -jr 0.25 -did 1 -go cnn -fr 2000 -rh 1 -bt 0.5 -tid 30003 > outputs/cifar10_fedmoco_20_30003.out 2>&1 &

# fedmgda
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 7 -go cnn -fr 1500 > outputs/cifar10_fedmgda_10.out 2>&1 &
# nohup python -u main.py -nc 5 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 7 -go cnn -fr 1500 > outputs/cifar10_fedmgda_5.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1500 -did 7 -go cnn -fr 1000 > outputs/cifar10_fedmgda_20.out 2>&1 &

# fedavg
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 5 -go cnn -fr 0 -bt 1 > outputs/cifar10_fedavg_10.out 2>&1 &
# nohup python -u main.py -nc 5 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -did 5 -go cnn -fr 0 -bt 1 > outputs/cifar10_fedavg_5.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1500 -did 5 -go cnn -fr 0 -bt 1 > outputs/cifar10_fedavg_20.out 2>&1 &

# fedmoco
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 6 -go cnn -fr 1500 -bt 0.5 > outputs/cifar10_fedmoco_10.out 2>&1 &
# nohup python -u main.py -nc 5 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 6 -go cnn -fr 1500 -bt 0.5 > outputs/cifar10_fedmoco_5.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMoCo -gr 1500 -did 6 -go cnn -fr 1000 -bt 0.5 > outputs/cifar10_fedmoco_20.out 2>&1 &

# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 3 -go cnn -fr 1500 -bt 0.01 > outputs/cifar10_fedmoco_10_001.out 2>&1 &
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 3 -go cnn -fr 1500 -bt 0.1 > outputs/cifar10_fedmoco_10_01.out 2>&1 &
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 3 -go cnn -fr 1500 -bt 0.5 > outputs/cifar10_fedmoco_10_05.out 2>&1 &
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 4 -go cnn -fr 1500 -bt 0.9 > outputs/cifar10_fedmoco_10_09.out 2>&1 &
# nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -did 4 -go cnn -fr 1500 -bt 0.99 > outputs/cifar10_fedmoco_10_099.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMoCo -gr 10 -did 6 -go cnn -fr 1500 -bt 0.5 > outputs/cifar10_fedmoco2.out 2>&1 &
# python main.py -nc 10 -data Cifar10 -m cnn -algo FedAvg -gr 1000 -did 6 -go cnn


# python main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1500 -did 6 -go cnn -fr 1000 -bt 0.5

