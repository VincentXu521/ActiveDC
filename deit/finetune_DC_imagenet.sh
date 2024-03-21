#!/bin/sh

# cd deit/

echo "[Start] ImageNet: 0.5"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/DC_sample/DC_ImageNet_EMD_sample_6405_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/imagenet-1k/emd_6405_DC_s0 &>> outputs_DC/imagenet-1k/DC_EMD_ImageNet_0.5.log

wait; echo "[Done] ImageNet: 0.5"



echo "[Start] ImageNet: 2"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/DC_sample/DC_ImageNet_EMD_sample_25623_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/imagenet-1k/emd_25623_DC_s0 &>> outputs_DC/imagenet-1k/DC_EMD_ImageNet_2.log

wait; echo "[Done] ImageNet: 2"



echo "[Start] ImageNet: 1"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/DC_sample/DC_ImageNet_EMD_sample_12811_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/imagenet-1k/emd_12811_DC_s0 &>> outputs_DC/imagenet-1k/DC_EMD_ImageNet_1.log

wait; echo "[Done] ImageNet: 1"



echo "[Start] ImageNet: 5"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
        --use_env main_DC.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/DC_sample/DC_ImageNet_EMD_sample_64058_indexs_labels.npz \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_DC/imagenet-1k/emd_64058_DC_s0 &>> outputs_DC/imagenet-1k/DC_EMD_ImageNet_5.log

wait; echo "[Done] ImageNet: 5"




<< EOF
bash finetune_DC_imagenet.sh 
-----------------------
... 

EOF

