#!/bin/sh
# cd data_selection && bash sample_tools/LR4FT.sh 


echo "LR start"
python sample_tools/LR4FT.py -d CIFAR10 --val --no_aug -p 0.1  &> DC_sample/logs_LR4FT/CIFAR10_0.1.log
wait; echo "CIFAR10: 0.1"
echo "----------------------------------"
python sample_tools/LR4FT.py -d CIFAR10 --val --no_aug -p 0.2  &> DC_sample/logs_LR4FT/CIFAR10_0.2.log
wait; echo "CIFAR10: 0.2"
echo "----------------------------------"
python sample_tools/LR4FT.py -d CIFAR10 --val --no_aug -p 0.5  &> DC_sample/logs_LR4FT/CIFAR10_0.5.log
wait; echo "CIFAR10: 0.5"
echo "----------------------------------"
python sample_tools/LR4FT.py -d CIFAR10 --val --no_aug -p 1  &> DC_sample/logs_LR4FT/CIFAR10_1.log
wait; echo "CIFAR10: 1"
echo "----------------------------------"
python sample_tools/LR4FT.py -d CIFAR10 --val --no_aug -p 2  &> DC_sample/logs_LR4FT/CIFAR10_2.log
wait; echo "CIFAR10: 2"
echo "=================================="


python sample_tools/LR4FT.py -d CIFAR100 --val --no_aug -p 1  &> DC_sample/logs_LR4FT/CIFAR100_1.log
wait; echo "CIFAR100: 1"
echo "----------------------------------"
python sample_tools/LR4FT.py -d CIFAR100 --val --no_aug -p 2  &> DC_sample/logs_LR4FT/CIFAR100_2.log
wait; echo "CIFAR100: 2"
echo "----------------------------------"
python sample_tools/LR4FT.py -d CIFAR100 --val --no_aug -p 5  &> DC_sample/logs_LR4FT/CIFAR100_5.log
wait; echo "CIFAR100: 5"
echo "----------------------------------"
python sample_tools/LR4FT.py -d CIFAR100 --val --no_aug -p 10  &> DC_sample/logs_LR4FT/CIFAR100_10.log
wait; echo "CIFAR100: 10"
echo "=================================="


python sample_tools/LR4FT.py -d ImageNet --val --no_aug -p 0.5  &> DC_sample/logs_LR4FT/ImageNet_0.5.log
wait; echo "ImageNet: 0.5"
echo "----------------------------------"
python sample_tools/LR4FT.py -d ImageNet --val --no_aug -p 1  &> DC_sample/logs_LR4FT/ImageNet_1.log
wait; echo "ImageNet: 1"
echo "----------------------------------"
python sample_tools/LR4FT.py -d ImageNet --val --no_aug -p 2  &> DC_sample/logs_LR4FT/ImageNet_2.log
wait; echo "ImageNet: 2"
echo "----------------------------------"
python sample_tools/LR4FT.py -d ImageNet --val --no_aug -p 5  &> DC_sample/logs_LR4FT/ImageNet_5.log
wait; echo "ImageNet: 5"
echo "=================================="


echo "LR end"
echo "path: data_selection/DC_sample/logs_LR4FT"
