#!/bin/sh

# cd deit/

# cat outputs_imagenet/EMD6500_s0/log_1.txt |grep -E "acc1.+," -o |awk '{sum += $2} END {print "Average = ", sum / NR}'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 0 \
        --clip-grad 2.0 \
        --data-set IMNETSUBSET \
        --subset_ids ../data_selection/DC_sample/ImageNet_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_100_sampleNum_6500.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_imagenet/EMD6500_s0 &> outputs_imagenet/EMD6500_s0/train_log.log


<< EOF
(active_finetune) wenshuai@node19:~/projects/ActiveFT_xu/deit$ cat outputs_imagenet/EMD6500_s0/log.txt |grep -E "acc1.+," -o
acc1": 80.02000167236328, "test_acc5": 95.060001953125, "epoch": 49,
acc1": 79.66000284423828, "test_acc5": 94.48000340576172, "epoch": 99,
acc1": 78.26000187988281, "test_acc5": 94.02000122070312, "epoch": 149,
acc1": 77.60000234375, "test_acc5": 93.54000086669922, "epoch": 199,
acc1": 77.90000081787109, "test_acc5": 93.66000297851562, "epoch": 249,
acc1": 78.14000234375, "test_acc5": 93.34000153808594, "epoch": 299,
acc1": 77.7400012084961, "test_acc5": 93.42000157470703, "epoch": 349,
acc1": 78.54000206298828, "test_acc5": 93.52000141601563, "epoch": 399,
acc1": 76.84000133056641, "test_acc5": 93.40000246582031, "epoch": 449,
acc1": 77.72000303955078, "test_acc5": 93.26000242919922, "epoch": 499,
acc1": 78.32000209960937, "test_acc5": 93.48000191650391, "epoch": 549,
acc1": 78.2000014038086, "test_acc5": 93.14000230712891, "epoch": 599,
acc1": 78.600003125, "test_acc5": 94.02000379638672, "epoch": 649,
acc1": 78.81999979248047, "test_acc5": 93.92000395507813, "epoch": 699,
acc1": 79.2400035522461, "test_acc5": 93.96000157470704, "epoch": 749,
acc1": 79.70000174560546, "test_acc5": 94.04000313720704, "epoch": 799,
acc1": 79.88000151367187, "test_acc5": 94.24000223388671, "epoch": 849,
acc1": 80.32000236816407, "test_acc5": 94.20000157470703, "epoch": 899,
acc1": 80.40000229492188, "test_acc5": 94.1400019165039, "epoch": 949,
acc1": 80.48000174560546, "test_acc5": 94.22000313720703, "epoch": 999
EOF





# Iter: 99, lr: 0.001000, loss: 3.246781
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --eval_interval 50 \
#         --data-set IMNETSUBSET \
#         --subset_ids ../data_selection/features/ImageNet_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_100_sampleNum_1300.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_imagenet/EMD1300_s0


# Iter: 299, lr: 0.001000, loss: 3.199055
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
#         --use_env main_eval_per50.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --eval_interval 50 \
#         --data-set IMNETSUBSET \
#         --subset_ids ../data_selection/features/ImageNet_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_1300.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_imagenet/EMD1300_s0_iter300


<< EOF
-----------------------
iter 100
... 
Test: Total time: 0:00:20 (2.8596 s / it)
* Acc@1 71.740 Acc@5 89.840 loss 1.962
Accuracy of the network on the 5000 test images: 71.7%
Max accuracy: 71.74%
Training time 1:27:39
-----------------------
iter 300
... 
Test: Total time: 0:00:20 (2.8619 s / it)
* Acc@1 72.380 Acc@5 90.920 loss 1.897
Accuracy of the network on the 5000 test images: 72.4%
Max accuracy: 72.50%
Training time 1:30:08
EOF

