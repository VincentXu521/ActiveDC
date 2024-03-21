# cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR100SUBSET \
        --subset_ids ../data_selection/features/CIFAR100_train_random_sampleNum_1000.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_cifar100/random1000_s0


<< EOF
-----------------------
seed 3405
... 
-----------------------
seed 0
... 
Test: Total time: 0:00:14 (1.0480 s / it)
* Acc@1 21.640 Acc@5 43.300 loss 3.663
Accuracy of the network on the 10000 test images: 21.6%
Max accuracy: 21.76%
Training time 1:01:25
EOF

