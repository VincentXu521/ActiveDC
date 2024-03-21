# cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --clip-grad 2.0 \
        --eval_interval 50 \
        --data-set CIFAR10SUBSET \
        --subset_ids ../data_selection/features/CIFAR10_train_random_sampleNum_1000.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_random


<< EOF
...
Test: Total time: 0:00:14 (1.0604 s / it)
* Acc@1 88.550 Acc@5 99.210 loss 0.425
Accuracy of the network on the 10000 test images: 88.6%
Max accuracy: 88.59%
Training time 0:55:33
EOF
