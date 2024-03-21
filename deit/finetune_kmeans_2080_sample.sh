# cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 --use_env main.py --clip-grad 2.0 --eval_interval 50 --data-set CIFAR10SUBSET --subset_ids ../data_selection/features/CIFAR10_train_KMeans_2080_sampleNum_1000.json --resume ckpt/dino_deitsmall16_pretrain.pth --output_dir outputs_kmeans_2080

