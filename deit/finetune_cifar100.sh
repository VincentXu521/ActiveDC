# cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR100SUBSET \
        --subset_ids ../data_selection/features/CIFAR100_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_1000.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_cifar100/EMD1000_s0


<< EOF
-----------------------
seed 3405
... 
Test: Total time: 0:00:14 (1.0183 s / it)
* Acc@1 21.470 Acc@5 43.400 loss 3.622
Accuracy of the network on the 10000 test images: 21.5%
Max accuracy: 21.48%
Training time 1:00:41
-----------------------
seed 0
... 
Test: Total time: 0:00:14 (1.0266 s / it)
* Acc@1 36.970 Acc@5 62.920 loss 3.014
Accuracy of the network on the 10000 test images: 37.0%
Max accuracy: 37.20%
Training time 1:00:43
EOF

