# cd deit/
# bash finetune_full-FT_DDP.sh


datasets=(CIFAR10 CIFAR100 ImageNet)
subsets=(CIFAR10SUBSET CIFAR100SUBSET IMNETSUBSET)
per_CIFAR10=(0.1 0.2 0.5 1 2)
per_CIFAR100=(1 2 5 10)
per_ImageNet=(0.5 1 2 5)

echo "full FT DDP start..."
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
                python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
                        --use_env main_eval_per50.py \
                        --seed 0 \
                        --epochs 300 \
                        --eval_interval 1 \
                        --clip-grad 2.0 \
                        --data-set $subset \
                        --subset_ids subset/${dataset}_${per}.json \
                        --resume ckpt/dino_deitsmall16_pretrain.pth \
                        --output_dir outputs_full_FT_ddp/${dataset}/per_${per} &> outputs_full_FT_ddp/${dataset}/${dataset}_${per}.log
                wait; echo $dataset $per "done."
        done
        echo "=========================================="
done
echo "full FT DDP done."

                        # --resume ckpt/dino_deitsmall16_pretrain.pth \
                        # --resume outputs_full_FT_ddp/${dataset}/per_${per}/checkpoint.pth \  # lr is too small



<< EOF
########################################################
epoch 50:
=========================================
cifar10, 0.1, 0.2, 0.5, 1, 2
-----------------------------------------
55.09, 70.32, 56.49, 69.90, 89.25 (epoch is not enough, not to final acc)

=========================================
cifar100, 1, 2, 5, 10
-----------------------------------------
8.93, 15.61, 31.76, 57.81 (epoch is not enough, not to final acc)

=========================================
imagenet, 0.5, 1, 2, 5
-----------------------------------------
25.70, 42.99, 57.23, 66.49 (epoch is not enough, not to final acc)
=========================================

EOF
