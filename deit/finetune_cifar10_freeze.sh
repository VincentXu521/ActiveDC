# cd deit/

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
#         --use_env main_freeze_encoder.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR10SUBSET \
#         --subset_ids ../data_selection/features/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_250.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_freeze/cifar10/emd250_s0


python main_freeze_simple.py \
        --seed 2 \
        --clip-grad 2.0 \
        --data-set CIFAR10SUBSET \
        --subset_ids ../data_selection/features/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_250.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_freeze/cifar10/emd250_not_ddp



<< EOF
bash finetune_cifar10_freeze.sh
-----------------------
... 
Test: Total time: 0:00:17 (1.2255 s / it)
* Acc@1 73.670 Acc@5 96.660 loss 0.854
Accuracy of the network on the 10000 test images: 73.7%
Max accuracy: 85.06%
Training time 0:50:48

########################################################
bash finetune_cifar10_freeze.sh
-----------------------
Test: Total time: 0:00:11 (1.6059 s / it)
* Acc@1 76.560 Acc@5 97.510 loss 0.764
Accuracy of the network on the 10000 test images: 76.6%
Max accuracy: 84.95%
Training time 0:49:39
EOF

