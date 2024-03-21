# cd deit/


python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --eval_interval 50 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/features/ImageNet_train_KMeans_3090_sampleNum_1300.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_imagenet/kmeans1300_s0


<< EOF
-----------------------
iter 100
... 

-----------------------
iter 300
... 
Test: Total time: 0:00:20 (2.9408 s / it)
* Acc@1 72.920 Acc@5 90.400 loss 1.863
Accuracy of the network on the 5000 test images: 72.9%
Max accuracy: 72.92%
Training time 1:29:17
EOF

