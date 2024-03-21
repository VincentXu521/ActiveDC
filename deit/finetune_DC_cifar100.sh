#!/bin/sh

# cd deit/

echo "[Start] CIFAR100: 1"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR100SUBSET \
        --subset_ids ../data_selection/DC_sample/DC_CIFAR100_EMD_sample_500_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/cifar100/emd_500_DC_s0 &> outputs_DC/cifar100/DC_EMD_CIFAR100_1.log

wait; echo "[Done] CIFAR100: 1"


echo "[Start] CIFAR100: 2"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR100SUBSET \
        --subset_ids ../data_selection/DC_sample/DC_CIFAR100_EMD_sample_1000_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/cifar100/emd_1000_DC_s0 &> outputs_DC/cifar100/DC_EMD_CIFAR100_2.log

wait; echo "[Done] CIFAR100: 2"


echo "[Start] CIFAR100: 5"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR100SUBSET \
        --subset_ids ../data_selection/DC_sample/DC_CIFAR100_EMD_sample_2500_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/cifar100/emd_2500_DC_s0 &> outputs_DC/cifar100/DC_EMD_CIFAR100_5.log

wait; echo "[Done] CIFAR100: 5"


echo "[Start] CIFAR100: 10"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR100SUBSET \
        --subset_ids ../data_selection/DC_sample/DC_CIFAR100_EMD_sample_5000_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/cifar100/emd_5000_DC_s0 &> outputs_DC/cifar100/DC_EMD_CIFAR100_10.log

wait; echo "[Done] CIFAR100: 10"






# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_DC.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR100SUBSET \
#         --subset_ids ../data_selection/features/CIFAR100_sample_1000_DC_indexs_labels.npz \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_DC/cifar100/kmeans_1000_DC_s0

<< EOF
bash finetune_DC_cifar100.sh 
-----------------------
... 
Test: Total time: 0:00:17 (1.2640 s / it)
* Acc@1 54.740 Acc@5 80.150 loss 2.340
Accuracy of the network on the 10000 test images: 54.7%
Max accuracy: 54.91%
Training time 1:38:07
EOF

