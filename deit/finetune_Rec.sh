# cd deit/

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR10SUBSET \
#         --subset_ids ../data_selection/features/CIFAR10_train_Rec_sampleNum_250.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_Rec/cifar10/rec250_s0


# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR100SUBSET \
#         --subset_ids ../data_selection/features/CIFAR100_train_Rec_sampleNum_1000.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_Rec/cifar100/rec1000_s0


python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/features/ImageNet_train_Rec_sampleNum_1300.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_Rec/imagenet/rec1300_s0


<< EOF
-----------------------
... 
* Acc@1 74.320 Acc@5 96.930 loss 0.929
Accuracy of the network on the 10000 test images: 74.3%
Max accuracy: 74.32%
Training time 0:48:05
...
* Acc@1 22.520 Acc@5 46.260 loss 3.600
Accuracy of the network on the 10000 test images: 22.5%
Max accuracy: 22.56%
Training time 0:55:55
...
* Acc@1 69.020 Acc@5 89.240 loss 2.045
Accuracy of the network on the 5000 test images: 69.0%
Max accuracy: 69.02%
Training time 1:35:31
EOF

