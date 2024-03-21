# cd deit/
# bash finetune_freeze.sh


datasets=(CIFAR10 CIFAR100 ImageNet)
subsets=(CIFAR10SUBSET CIFAR100SUBSET IMNETSUBSET)
per_CIFAR10=(0.1 0.2 0.5 1 2)
per_CIFAR100=(1 2 5 10)
per_ImageNet=(0.5 1 2 5)

echo "Linear Probing start..."
for i in {0..2}
do
        dataset=${datasets[${i}]}
        subset=${subsets[${i}]}
        echo $dataset $subset
        if [ "$dataset" == "CIFAR10" ]; then
                data_per=${per_CIFAR10[@]}
        elif [ "$dataset" == "CIFAR100" ]; then
                data_per=${per_CIFAR100[@]}
        else
                data_per=${per_ImageNet[@]}
        fi
        echo $dataset: $data_per
        # for per in ${data_per[@]}
        for per in $data_per
        do
                echo "----------------------------------"
                echo $dataset $per "start..."
                python main_freeze_simple.py \
                        --seed 2 \
                        --epochs 50 \
                        --eval_interval 1 \
                        --clip-grad 2.0 \
                        --data-set $subset \
                        --subset_ids subset/${dataset}_${per}.json \
                        --resume ckpt/dino_deitsmall16_pretrain.pth \
                        --output_dir outputs_freeze/${dataset}/per_${per} &> outputs_freeze/${dataset}/${dataset}_${per}.log
                wait; echo $dataset $per "done."
        done
        echo "=========================================="
done
echo "Linear Probing done."




<< EOF
########################################################
cifar10, 0.1, 0.2, 0.5, 1, 2
-----------------------------------------
76.60, 82.58, 86.72, 88.25, 91.17

=========================================
cifar100, 1, 2, 5, 10
-----------------------------------------
48.45, 59.76, 67.39, 72.64

=========================================
imagenet, 0.5, 1, 2, 5
-----------------------------------------
55.91, 60.59, 63.86, 68.36
=========================================

EOF



# single GPU type:

# echo "start..."
# python main_freeze_simple.py \
#         --seed 2 \
#         --clip-grad 2.0 \
#         --data-set CIFAR10SUBSET \
#         --subset_ids subset/CIFAR10_0.5.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_freeze/CIFAR10/per_0.5 &> outputs_freeze/CIFAR10/CIFAR10_0.5.log
# wait; echo "done."



# DDP type:

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
#         --use_env main_freeze_encoder.py \
#         --seed 0 \
#         --clip-grad 2.0 \
#         --data-set CIFAR10SUBSET \
#         --subset_ids ../data_selection/features/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_250.json \
#         --resume ckpt/dino_deitsmall16_pretrain.pth \
#         --output_dir outputs_freeze/cifar10/emd250_s0


