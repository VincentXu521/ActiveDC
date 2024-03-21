# cd deit/

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 3405 \
#         --clip-grad 2.0 \
#         --data-set CIFAR100SUBSET \
#         --subset_ids ../data_selection/features/CIFAR100_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_1000_sampleNum_1000.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_cifar100/EMD1000_s3405_iter1000



python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR100SUBSET \
        --subset_ids ../data_selection/features/CIFAR100_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_800.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_cifar100/EMD800_s0



<< EOF
iter_1000
-----------------------
seed 3405
... 
Test: Total time: 0:00:14 (1.0325 s / it)
* Acc@1 22.020 Acc@5 45.180 loss 3.575
Accuracy of the network on the 10000 test images: 22.0%
Max accuracy: 22.24%
Training time 1:00:06
-----------------------
seed 0
... 
Test: Total time: 0:00:14 (1.0601 s / it)
* Acc@1 33.260 Acc@5 58.330 loss 3.162
Accuracy of the network on the 10000 test images: 33.3%
Max accuracy: 33.45%
Training time 1:00:55

sample 800
-----------------------
seed 0
...
Test: Total time: 0:00:14 (1.0319 s / it)
* Acc@1 28.350 Acc@5 53.620 loss 3.350
Accuracy of the network on the 10000 test images: 28.4%
Max accuracy: 28.46%
Training time 1:00:59
EOF

