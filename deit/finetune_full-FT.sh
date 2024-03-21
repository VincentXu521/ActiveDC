# cd deit/
# bash finetune_full-FT.sh


datasets=(CIFAR10 CIFAR100 ImageNet)
subsets=(CIFAR10SUBSET CIFAR100SUBSET IMNETSUBSET)
per_CIFAR10=(0.1 0.2 0.5 1 2)
per_CIFAR100=(1 2 5 10)
per_ImageNet=(0.5 1 2 5)

echo "full FT start..."
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
                python main_single.py \
                        --seed 2 \
                        --epochs 50 \
                        --eval_interval 1 \
                        --clip-grad 2.0 \
                        --data-set $subset \
                        --subset_ids subset/${dataset}_${per}.json \
                        --resume ckpt/dino_deitsmall16_pretrain.pth \
                        --output_dir outputs_full_FT/${dataset}/per_${per} &> outputs_full_FT/${dataset}/${dataset}_${per}.log
                wait; echo $dataset $per "done."
        done
        echo "=========================================="
done
echo "full FT done."




<< EOF
########################################################
cifar10, 0.1, 0.2, 0.5, 1, 2
-----------------------------------------


=========================================
cifar100, 1, 2, 5, 10
-----------------------------------------


=========================================
imagenet, 0.5, 1, 2, 5
-----------------------------------------

=========================================

EOF
