# cd deit/

# Iter: 99, lr: 0.001000, loss: 3.246781
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --eval_interval 50 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/features/ImageNet_train_random_sampleNum_1300.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_imagenet/random1300_s0


<< EOF
Test: Total time: 0:00:24 (3.5321 s / it)
* Acc@1 68.220 Acc@5 88.220 loss 2.080
Accuracy of the network on the 5000 test images: 68.2%
Max accuracy: 68.30%
Training time 1:24:50
EOF

