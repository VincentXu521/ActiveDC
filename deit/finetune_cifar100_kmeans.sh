# cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 3405 \
        --clip-grad 2.0 \
        --data-set CIFAR100SUBSET \
        --subset_ids ../data_selection/features/CIFAR100_train_KMeans_3090_sampleNum_1000.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_cifar100/KMeans1000_s3405


<< EOF
-----------------------
seed 3405
... 
Test: Total time: 0:00:14 (1.0169 s / it)
* Acc@1 18.850 Acc@5 39.360 loss 3.815
Accuracy of the network on the 10000 test images: 18.9%
Max accuracy: 18.96%
Training time 1:00:49
-----------------------
seed 0
... 
Test: Total time: 0:00:14 (1.0056 s / it)
* Acc@1 40.470 Acc@5 65.450 loss 2.900
Accuracy of the network on the 10000 test images: 40.5%
Max accuracy: 40.57%
Training time 1:01:00
EOF

