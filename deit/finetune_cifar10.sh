#!/bin/sh

# cd deit/

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR10SUBSET \
#         --subset_ids ../data_selection/DC_sample/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_50.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_cifar10/EMD50_s0

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR10SUBSET \
#         --subset_ids ../data_selection/DC_sample/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_100.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_cifar10/EMD100_s0

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR10SUBSET \
#         --subset_ids ../data_selection/DC_sample/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_250.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_cifar10/EMD250_s0


# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR10SUBSET \
#         --subset_ids ../data_selection/DC_sample/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_500.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_cifar10/EMD500_s0


# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR10SUBSET \
#         --subset_ids ../data_selection/DC_sample/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_1000.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_cifar10/EMD1000_s0


<< EOF
-----------------------
seed 0, sample 250
... 
Accuracy of the network on the 10000 test images: 79.3%
Max accuracy: 79.35%
Training time 0:50:48
...
seed 0, sample 500
Accuracy of the network on the 10000 test images: 86.5%
Max accuracy: 86.53%
Training time 0:50:49
...
seed 0, sample 1000

EOF

