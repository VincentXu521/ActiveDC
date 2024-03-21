# cd deit/


# echo "[Start] ImageNet: 0.5"
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set IMNETSUBSET \
#         --subset_ids ../data_selection/features/ImageNet_train_KMeans_3090_sampleNum_6405.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_kmeans_3090/kmeans_6405_s0 &> outputs_kmeans_3090/train_log_0.5.log

# wait; echo "[Done] ImageNet: 0.5"


# echo "[Start] ImageNet: 1"
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set IMNETSUBSET \
#         --subset_ids ../data_selection/features/ImageNet_train_KMeans_3090_sampleNum_12811.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_kmeans_3090/kmeans_12811_s0 &> outputs_kmeans_3090/train_log_1.log

# wait; echo "[Done] ImageNet: 1"


# echo "[Start] ImageNet: 2"
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set IMNETSUBSET \
#         --subset_ids ../data_selection/features/ImageNet_train_KMeans_3090_sampleNum_25623.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_kmeans_3090/kmeans_25623_s0 &> outputs_kmeans_3090/train_log_2.log

# wait; echo "[Done] ImageNet: 2"


echo "[Start] ImageNet: 5"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/features/ImageNet_train_KMeans_3090_sampleNum_64058.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_kmeans_3090/kmeans_64058_s0 &> outputs_kmeans_3090/train_log_5.log

wait; echo "[Done] ImageNet: 5"






<< EOF
kmeans imagenet-1k 0.5%: 23min;
kmeans imagenet-1k 1%: **min;
kmeans imagenet-1k 2%: 51min;
...
Accuracy of the network on the 50000 test images: 37.1%
Max accuracy: 37.14%
Training time 5:46:33
---
Accuracy of the network on the 50000 test images: 50.8%
Max accuracy: 50.77%
Training time 7:24:07
---
Accuracy of the network on the 50000 test images: 55.7%
Max accuracy: 55.70%
Training time 12:06:15
---
Accuracy of the network on the 50000 test images: 60.4%
Max accuracy: 62.24%
Training time 1 day, 2:37:33
EOF
