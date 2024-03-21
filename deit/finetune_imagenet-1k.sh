#!/bin/sh

# cd deit/

# cat outputs_imagenet-1k/EMD6405_s0/log.txt |grep -E "acc1.+," -o
# cat outputs_imagenet-1k/EMD6405_s0/log_1.txt |grep -E "acc1.+," -o |awk '{sum += $2} END {print "Average = ", sum / NR}'

echo "[Start] ImageNet: 0.5"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/DC_sample/ImageNet_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_100_sampleNum_6405.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_imagenet-1k/EMD6405_s0 &> outputs_imagenet-1k/train_log_0.5.log

wait; echo "[Done] ImageNet: 0.5"


echo "[Start] ImageNet: 2"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/DC_sample/ImageNet_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_100_sampleNum_25623.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_imagenet-1k/EMD25623_s0 &> outputs_imagenet-1k/train_log_2.log

wait; echo "[Done] ImageNet: 2"




<< EOF
percent 0.5%
...
* Acc@1 36.788 Acc@5 57.850 loss 4.247
Accuracy of the network on the 50000 test images: 36.8%
Max accuracy: 36.82%
Training time 7:35:14
---
percent 2%
...
* Acc@1 54.238 Acc@5 75.136 loss 2.915
Accuracy of the network on the 50000 test images: 54.2%
Max accuracy: 54.24%
Training time 16:13:23
EOF
