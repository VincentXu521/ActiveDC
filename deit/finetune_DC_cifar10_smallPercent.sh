#!/bin/sh

# cd deit/

echo "[Start] CIFAR10: 0.1"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR10SUBSET \
        --subset_ids ../data_selection/DC_sample/DC_CIFAR10_EMD_sample_50_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/cifar10/emd_50_DC_s0 &> outputs_DC/cifar10/DC_EMD_CIFAR10_0.1.log

wait; echo "[Done] CIFAR10: 0.1"


echo "[Start] CIFAR10: 0.2"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR10SUBSET \
        --subset_ids ../data_selection/DC_sample/DC_CIFAR10_EMD_sample_100_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/cifar10/emd_100_DC_s0 &> outputs_DC/cifar10/DC_EMD_CIFAR10_0.2.log

wait; echo "[Done] CIFAR10: 0.2"


<< EOF
bash finetune_DC_cifar10_smallPercent.sh
-----------------------
... 
percent 0.1%:
Accuracy of the network on the 10000 test images: 60.8%
Max accuracy: 61.36%
Training time 0:44:52
...
percent 0.2%:
Accuracy of the network on the 10000 test images: 72.6%
Max accuracy: 73.16%
Training time 0:48:03
EOF

