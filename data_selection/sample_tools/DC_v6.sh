#!/bin/sh

# cd data_selection && bash sample_tools/DC_v6.sh 

# python sample_tools/ActiveFT_CIFAR.py --feature_path features/CIFAR10_train.npy --output_dir DC_sample --percent (0.5, 1, 2)
# python sample_tools/ActiveFT_CIFAR.py --feature_path features/CIFAR100_train.npy --output_dir DC_sample --percent (1, 2, 5, 10)
# python sample_tools/ActiveFT_ImageNet.py --feature_path features/ImageNet_train.npy --output_dir DC_sample --percent (1, 5)

echo "DC sample start"
python sample_tools/DC_v6_sample.py -b CIFAR10 -d CIFAR10 --val --gpu -un -p 0.5  &> DC_sample/logs/CIFAR10_0.5.log
wait; echo "CIFAR10: 0.5"
python sample_tools/DC_v6_sample.py -b CIFAR10 -d CIFAR10 --val --gpu -un -p 1    &> DC_sample/logs/CIFAR10_1.log
wait; echo "CIFAR10: 1"
python sample_tools/DC_v6_sample.py -b CIFAR10 -d CIFAR10 --val --gpu -un -p 2    &> DC_sample/logs/CIFAR10_2.log
wait; echo "CIFAR10: 2"
python sample_tools/DC_v6_sample.py -b CIFAR100 -d CIFAR100 --val --gpu -un -p 1  &> DC_sample/logs/CIFAR100_1.log
wait; echo "CIFAR100: 1"
python sample_tools/DC_v6_sample.py -b CIFAR100 -d CIFAR100 --val --gpu -un -p 2  &> DC_sample/logs/CIFAR100_2.log
wait; echo "CIFAR100: 2"
python sample_tools/DC_v6_sample.py -b CIFAR100 -d CIFAR100 --val --gpu -un -p 5  &> DC_sample/logs/CIFAR100_5.log
wait; echo "CIFAR100: 5"
python sample_tools/DC_v6_sample.py -b CIFAR100 -d CIFAR100 --val --gpu -un -p 10 &> DC_sample/logs/CIFAR100_10.log
wait; echo "CIFAR100: 10"
python sample_tools/DC_v6_sample.py -b ImageNet -d ImageNet --val --gpu -un -p 1  &> DC_sample/logs/ImageNet_1.log
wait; echo "ImageNet: 1"
python sample_tools/DC_v6_sample.py -b ImageNet -d ImageNet --val --gpu -un -p 5  &> DC_sample/logs/ImageNet_5.log
wait; echo "ImageNet: 5"
echo "DC sample end"

# python sample_tools/DC_v6_sample.py -b CIFAR10 -d CIFAR10 -p 0.5 --val --gpu -r features/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_250.json -un
# python sample_tools/DC_v6_sample.py -b CIFAR10 -d CIFAR10 -p 1 --val --gpu -r features/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_500.json -un
# python sample_tools/DC_v6_sample.py -b CIFAR10 -d CIFAR10 -p 2 --val --gpu -r features/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_1000.json -un
# python sample_tools/DC_v6_sample.py -b CIFAR100 -d CIFAR100 -p 1 --val --gpu -r features/CIFAR100_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_500.json -un
# python sample_tools/DC_v6_sample.py -b CIFAR100 -d CIFAR100 -p 2 --val --gpu -r features/CIFAR100_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_1000.json -un
# python sample_tools/DC_v6_sample.py -b CIFAR100 -d CIFAR100 -p 5 --val --gpu -r features/CIFAR100_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_2500.json -un
# python sample_tools/DC_v6_sample.py -b CIFAR100 -d CIFAR100 -p 10 --val --gpu -r features/CIFAR100_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_5000.json -un
# python sample_tools/DC_v6_sample.py -b ImageNet -d ImageNet -p 1 --val --gpu -r features/ImageNet_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_100_sampleNum_1300.json -un
# python sample_tools/DC_v6_sample.py -b ImageNet -d ImageNet -p 5 --val --gpu -r features/ImageNet_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_100_sampleNum_6500.json -un
