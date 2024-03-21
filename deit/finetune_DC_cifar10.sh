#!/bin/sh

# cd deit/

echo "[Start] CIFAR10: 0.5"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR10SUBSET \
        --subset_ids ../data_selection/DC_sample/DC_CIFAR10_EMD_sample_250_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/cifar10/emd_250_DC_s0 &> outputs_DC/cifar10/DC_EMD_CIFAR10_0.5.log

wait; echo "[Done] CIFAR10: 0.5"


echo "[Start] CIFAR10: 1"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR10SUBSET \
        --subset_ids ../data_selection/DC_sample/DC_CIFAR10_EMD_sample_500_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/cifar10/emd_500_DC_s0 &> outputs_DC/cifar10/DC_EMD_CIFAR10_1.log

wait; echo "[Done] CIFAR10: 1"


echo "[Start] CIFAR10: 2"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR10SUBSET \
        --subset_ids ../data_selection/DC_sample/DC_CIFAR10_EMD_sample_1000_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/cifar10/emd_1000_DC_s0 &> outputs_DC/cifar10/DC_EMD_CIFAR10_2.log

wait; echo "[Done] CIFAR10: 2"



# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_DC.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR10SUBSET \
#         --subset_ids ../data_selection/features/CIFAR10_sample_250_DC_indexs_labels.npz \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_DC/cifar10/kmeans_250_DC_s0

<< EOF
bash finetune_DC_cifar10.sh
-----------------------
... 
Test: Total time: 0:00:16 (1.2024 s / it)
* Acc@1 85.660 Acc@5 98.930 loss 0.545
Accuracy of the network on the 10000 test images: 85.7%
Max accuracy: 85.82%
Training time 0:57:24
EOF

