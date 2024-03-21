# cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 \
        --use_env main.py \
        --eval \
        --dist-eval \
        --data-set CIFAR10 \
        --resume outputs/best_checkpoint.pth


<< EOF
(active_finetune) wenshuai@node19:~/projects/ActiveFT_xu/deit$ bash eval_activeFT.sh 
/gpfsdata/home/wenshuai/miniconda3/envs/active_finetune/lib/python3.9/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
| distributed init (rank 0): env://
| distributed init (rank 1): env://
Namespace(batch_size=256, epochs=1000, model='deit_small_patch16_224', input_size=224, drop=0.0, drop_path=0.1, model_ema=True, model_ema_decay=0.99996, model_ema_force_cpu=False, opt='SGD', opt_eps=1e-08, opt_betas=None, clip_grad=None, momentum=0.9, weight_decay=0.0001, sched='cosine', lr=0.01, lr_noise=None, lr_noise_pct=0.67, lr_noise_std=1.0, warmup_lr=1e-06, min_lr=1e-05, decay_epochs=30, warmup_epochs=5, cooldown_epochs=10, patience_epochs=10, decay_rate=0.1, color_jitter=0.0, aa=None, smoothing=0.1, train_interpolation='bicubic', repeated_aug=True, reprob=0, remode='pixel', recount=1, resplit=False, mixup=0, cutmix=0, cutmix_minmax=None, mixup_prob=0, mixup_switch_prob=0.5, mixup_mode='batch', teacher_model='regnety_160', teacher_path='', distillation_type='none', distillation_alpha=0.5, distillation_tau=1.0, finetune='', data_path='data', data_set='CIFAR10', inat_category='name', output_dir='', device='cuda', seed=0, resume='outputs/best_checkpoint.pth', start_epoch=0, eval=True, dist_eval=True, num_workers=16, pin_mem=True, world_size=2, dist_url='env://', subset_ids=None, eval_interval=1, rank=0, gpu=0, distributed=True, dist_backend='nccl')
Files already downloaded and verified
Files already downloaded and verified
Creating model: deit_small_patch16_224
number of params: 21669514
Test:  [0/7]  eta: 0:01:13  loss: 0.3814 (0.3814)  acc1: 90.2344 (90.2344)  acc5: 99.4792 (99.4792)  time: 10.4716  data: 9.2218  max mem: 2755
Test:  [6/7]  eta: 0:00:01  loss: 0.3946 (0.3982)  acc1: 90.1042 (90.0000)  acc5: 99.4792 (99.2600)  time: 1.8541  data: 1.3386  max mem: 2755
Test: Total time: 0:00:13 (1.9140 s / it)
* Acc@1 90.220 Acc@5 99.330 loss 0.391
Accuracy of the network on the 10000 test images: 90.2%
EOF
