# cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set CIFAR100SUBSET \
        --subset_ids ../data_selection/features/CIFAR100_train_EMD800_UN200_comb_sampleNum_1000.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_cifar100/comb_EMD800_BvSB200_s0


<< EOF
-----------------------
seed 3405
... 

-----------------------
seed 0
... 
Test: Total time: 0:00:14 (1.0195 s / it)
* Acc@1 34.020 Acc@5 59.050 loss 3.153
Accuracy of the network on the 10000 test images: 34.0%
Max accuracy: 34.14%
Training time 1:00:32
EOF

