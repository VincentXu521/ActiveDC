# cd deit/


python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/features/ImageNet_train_PCA_sorted_sampleNum_1300.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_imagenet/pca1300_s0


<< EOF
-----------------------
... 
Test: Total time: 0:00:21 (3.1269 s / it)
* Acc@1 63.300 Acc@5 86.640 loss 2.551
Accuracy of the network on the 5000 test images: 63.3%
Max accuracy: 63.30%
Training time 1:33:38
EOF

