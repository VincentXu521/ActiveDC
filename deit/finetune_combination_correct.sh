# cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main_eval_per50.py \
        --seed 3405 \
        --clip-grad 2.0 \
        --eval_interval 50 \
        --data-set CIFAR10SUBSET \
        --subset_ids ../data_selection/features/CIFAR10_train_comb_correct_sampleNum_1000.json \
        --resume ckpt/dino_deitsmall16_pretrain.pth \
        --output_dir outputs_combination_correct/EMD800_BvSB200_shuffle_3405


<< EOF
finetune 800 sample ids, ../data_selection/features/CIFAR10_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_800.json
...
Test: Total time: 0:00:14 (1.0347 s / it)
* Acc@1 88.570 Acc@5 99.080 loss 0.430
Accuracy of the network on the 10000 test images: 88.6%
Max accuracy: 88.57%
Training time 1:02:26
------------------------------------------------------------------
finetune 1000 sample ids, ../data_selection/features/CIFAR10_train_comb_correct_sampleNum_1000.json, seed=0
...
Test: Total time: 0:00:14 (1.0539 s / it)
* Acc@1 90.290 Acc@5 99.510 loss 0.375
Accuracy of the network on the 10000 test images: 90.3%
Max accuracy: 90.29%
Training time 1:02:04
------------------------------------------------------------------
finetune 1000 sample ids, ../data_selection/features/CIFAR10_train_comb_correct_sampleNum_1000.json, seed=3407
...
Test: Total time: 0:00:14 (1.0512 s / it)
* Acc@1 89.950 Acc@5 99.420 loss 0.383
Accuracy of the network on the 10000 test images: 90.0%
Max accuracy: 90.08%
Training time 1:02:14
------------------------------------------------------------------
finetune 1000 sample ids, ../data_selection/features/CIFAR10_train_comb_correct_sampleNum_1000.json, seed=3405
...
Test: Total time: 0:00:14 (1.0481 s / it)
* Acc@1 90.360 Acc@5 99.410 loss 0.376
Accuracy of the network on the 10000 test images: 90.4%
Max accuracy: 90.42%
Training time 1:02:29
EOF
