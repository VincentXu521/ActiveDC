# cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --clip-grad 2.0 \
        --eval_interval 50 \
        --data-set CIFAR10SUBSET \
        --subset_ids ../data_selection/features/CIFAR10_train_comb_sampleNum_1000.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_combination


<< EOF
...
Test: Total time: 0:00:16 (1.1664 s / it)
* Acc@1 89.730 Acc@5 99.320 loss 0.398
Accuracy of the network on the 10000 test images: 89.7%
Max accuracy: 89.85%
Training time 1:05:06
EOF
