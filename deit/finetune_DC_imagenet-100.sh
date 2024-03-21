#!/bin/sh

# cd deit/


# echo "[Start] ImageNet: 5"
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
#         --use_env main_DC.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set IMNETSUBSET \
#         --subset_ids ../data_selection/DC_sample/DC_ImageNet_EMD_sample_6500_indexs_labels.npz \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_DC/imagenet/emd_6500_DC_s0 &> outputs_DC/imagenet/DC_EMD_ImageNet_5.log

# wait; echo "[Done] ImageNet: 5"



# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_DC.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set IMNETSUBSET \
#         --subset_ids ../data_selection/features/ImageNet_sample_1300_DC_indexs_labels.npz \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_DC/imagenet/kmeans_1300_DC_s0

<< EOF
bash finetune_DC_imagenet.sh 
-----------------------
... 
Test: Total time: 0:00:23 (3.3541 s / it)
* Acc@1 77.740 Acc@5 93.320 loss 1.618
Accuracy of the network on the 5000 test images: 77.7%
Max accuracy: 78.04%
Training time 1:59:47
EOF

