#!/bin/bash
nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMGDA -gr 1000   -did 0 -go cnn -fr 0 -tid 10000 > outputs/cifar10_fedavg_10_10000.out 2>&1 &

nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMGDA -gr 1000  -did 1 -go cnn -fr 2000 -bt 1 -tid 20000 > outputs/cifar10_fedmgda_10_20000.out 2>&1 &

nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMoCo -gr 1000  -did 2 -go cnn -fr 2000 -bt 0.1 -tid 30000 > outputs/cifar10_fedmoco_10_30000.out 2>&1 &

nohup python -u main.py -nc 10 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -stw 0.1  -did 3 -go cnn -fr 0 -tid 40000 > outputs/cifar10_fedstatic_10_40000.out 2>&1 &
# resnet version
# nohup python -u main.py -nc 10 -data Cifar10 -m resnet -algo FedMGDA -gr 1000   -did 0 -go cnn -fr 0 -tid 11000 > outputs/cifar10_fedavg_10_11000.out 2>&1 &

# nohup python -u main.py -nc 10 -data Cifar10 -m resnet -algo FedMGDA -gr 1000  -did 1 -go cnn -fr 2000 -bt 1 -tid 21000 > outputs/cifar10_fedmgda_10_21000.out 2>&1 &

# nohup python -u main.py -nc 10 -data Cifar10 -m resnet -algo FedMoCo -gr 1000  -did 2 -go cnn -fr 2000 -bt 0.5 -tid 31000 > outputs/cifar10_fedmoco_10_31000.out 2>&1 &
# join ratio experiment 
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -jr 0.5 -did 7 -go cnn -fr 2000 -tid 20001 > outputs/cifar10_fedmgda_20.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -jr 0.25 -did 7 -go cnn -fr 2000 -tid 20002 > outputs/cifar10_fedmgda_5.out 2>&1 &

# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -jr 0.5 -did 5 -go cnn -fr 0 -bt 1 -tid 10001 > outputs/cifar10_fedavg_20.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMGDA -gr 1000 -jr 0.25 -did 5 -go cnn -fr 0 -bt 1 -tid 10002 > outputs/cifar10_fedavg_5.out 2>&1 &

# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -jr 0.5 -did 6 -go cnn -fr 2000 -bt 0.5 -tid 30001 > outputs/cifar10_fedmoco_20.out 2>&1 &
# nohup python -u main.py -nc 20 -data Cifar10 -m cnn -algo FedMoCo -gr 1000 -jr 0.25 -did 6 -go cnn -fr 2000 -bt 0.5 -tid 30002 > outputs/cifar10_fedmoco_5.out 2>&1 &

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

