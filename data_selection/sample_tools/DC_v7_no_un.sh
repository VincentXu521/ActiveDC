#!/bin/sh
# cd data_selection && bash sample_tools/DC_v7_no_un.sh 

echo "DC sample start"
python sample_tools/DC_v7_sample.py -b CIFAR10 -d CIFAR10 --val --gpu -p 0.5  &> DC_sample/logs/no_un_CIFAR10_0.5.log
wait; echo "CIFAR10: 0.5"
python sample_tools/DC_v7_sample.py -b CIFAR10 -d CIFAR10 --val --gpu -p 1    &> DC_sample/logs/no_un_CIFAR10_1.log
wait; echo "CIFAR10: 1"
python sample_tools/DC_v7_sample.py -b CIFAR10 -d CIFAR10 --val --gpu -p 2    &> DC_sample/logs/no_un_CIFAR10_2.log
wait; echo "CIFAR10: 2"
python sample_tools/DC_v7_sample.py -b CIFAR100 -d CIFAR100 --val --gpu -p 1  &> DC_sample/logs/no_un_CIFAR100_1.log
wait; echo "CIFAR100: 1"
python sample_tools/DC_v7_sample.py -b CIFAR100 -d CIFAR100 --val --gpu -p 2  &> DC_sample/logs/no_un_CIFAR100_2.log
wait; echo "CIFAR100: 2"
python sample_tools/DC_v7_sample.py -b CIFAR100 -d CIFAR100 --val --gpu -p 5  &> DC_sample/logs/no_un_CIFAR100_5.log
wait; echo "CIFAR100: 5"
python sample_tools/DC_v7_sample.py -b CIFAR100 -d CIFAR100 --val --gpu -p 10 &> DC_sample/logs/no_un_CIFAR100_10.log
wait; echo "CIFAR100: 10"
python sample_tools/DC_v7_sample.py -b ImageNet -d ImageNet --val --gpu -p 1  &> DC_sample/logs/no_un_ImageNet_1.log
wait; echo "ImageNet: 1"
python sample_tools/DC_v7_sample.py -b ImageNet -d ImageNet --val --gpu -p 5  &> DC_sample/logs/no_un_ImageNet_5.log
wait; echo "ImageNet: 5"
echo "DC sample end"
