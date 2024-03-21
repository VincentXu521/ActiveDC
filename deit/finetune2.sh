# cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 3405 \
        --clip-grad 2.0 \
        --eval_interval 50 \
        --data-set CIFAR10SUBSET \
        --subset_ids ../data_selection/features/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_1000.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs2


<< EOF
-----------------------
seed 3405
... 
Test: Total time: 0:00:14 (1.0487 s / it)
* Acc@1 89.140 Acc@5 99.270 loss 0.409
Accuracy of the network on the 10000 test images: 89.1%
Max accuracy: 89.14%
Training time 1:02:31
EOF

