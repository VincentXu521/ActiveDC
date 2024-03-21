# cd deit/

python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 \
        --use_env main_eval.py \
        --eval \
        --dist-eval \
        --data-set CIFAR10 \
        --resume outputs/best_checkpoint.pth


<< EOF
----------------------------------
dataset_val, --nproc_per_node=2
----------------------------------

(active_finetune) wenshuai@node19:~/projects/ActiveFT_xu/deit$ bash eval_oneByOne.sh 
...
Creating model: deit_small_patch16_224
number of params: 21669514
Test:  [0/7]  eta: 0:01:04  loss: 0.3814 (0.3814)  acc1: 90.2344 (90.2344)  acc5: 99.4792 (99.4792)  time: 9.1533  data: 8.2858  max mem: 2755
Test:  [6/7]  eta: 0:00:01  loss: 0.3946 (0.3982)  acc1: 90.1042 (90.0000)  acc5: 99.4792 (99.2600)  time: 1.6107  data: 1.1838  max mem: 2755
Test: Total time: 0:00:11 (1.6609 s / it)
* Acc@1 90.220 Acc@5 99.330 loss 0.391
Accuracy of the network on the 10000 test images: 90.2%


----------------------------------
dataset_train, --nproc_per_node=1
----------------------------------

(active_finetune) wenshuai@node19:~/projects/ActiveFT_xu/deit$ bash eval_oneByOne.sh 
...
| distributed init (rank 0): env://
Namespace(batch_size=256, epochs=1000, model='deit_small_patch16_224', input_size=224, drop=0.0, drop_path=0.1, model_ema=True, model_ema_decay=0.99996, model_ema_force_cpu=False, opt='SGD', opt_eps=1e-08, opt_betas=None, clip_grad=None, momentum=0.9, weight_decay=0.0001, sched='cosine', lr=0.01, lr_noise=None, lr_noise_pct=0.67, lr_noise_std=1.0, warmup_lr=1e-06, min_lr=1e-05, decay_epochs=30, warmup_epochs=5, cooldown_epochs=10, patience_epochs=10, decay_rate=0.1, color_jitter=0.0, aa=None, smoothing=0.1, train_interpolation='bicubic', repeated_aug=True, reprob=0, remode='pixel', recount=1, resplit=False, mixup=0, cutmix=0, cutmix_minmax=None, mixup_prob=0, mixup_switch_prob=0.5, mixup_mode='batch', teacher_model='regnety_160', teacher_path='', distillation_type='none', distillation_alpha=0.5, distillation_tau=1.0, finetune='', data_path='data', data_set='CIFAR10', inat_category='name', output_dir='', device='cuda', seed=0, resume='outputs/best_checkpoint.pth', start_epoch=0, eval=True, dist_eval=True, num_workers=16, pin_mem=True, world_size=1, dist_url='env://', subset_ids=None, eval_interval=1, rank=0, gpu=0, distributed=True, dist_backend='nccl')
Files already downloaded and verified
Files already downloaded and verified
Creating model: deit_small_patch16_224
number of params: 21669514
Test:  [ 0/66]  eta: 0:12:05  loss: 0.6347 (0.6347)  acc1: 82.2917 (82.2917)  acc5: 97.5260 (97.5260)  time: 10.9990  data: 10.1883  max mem: 2753
Test:  [10/66]  eta: 0:01:34  loss: 0.6310 (0.6253)  acc1: 82.6823 (82.4692)  acc5: 97.5260 (97.6563)  time: 1.6797  data: 1.1701  max mem: 2753
Test:  [20/66]  eta: 0:00:51  loss: 0.6310 (0.6309)  acc1: 82.4219 (82.3041)  acc5: 97.5260 (97.6439)  time: 0.6199  data: 0.1344  max mem: 2753
Test:  [30/66]  eta: 0:00:32  loss: 0.6168 (0.6213)  acc1: 82.6823 (82.6193)  acc5: 97.5260 (97.6142)  time: 0.4894  data: 0.0005  max mem: 2753
Test:  [40/66]  eta: 0:00:20  loss: 0.5999 (0.6181)  acc1: 82.8125 (82.7268)  acc5: 97.7865 (97.6626)  time: 0.4771  data: 0.0121  max mem: 2753
Test:  [50/66]  eta: 0:00:11  loss: 0.6065 (0.6175)  acc1: 82.6823 (82.7410)  acc5: 97.7865 (97.6869)  time: 0.4247  data: 0.0118  max mem: 2753
Test:  [60/66]  eta: 0:00:03  loss: 0.6072 (0.6168)  acc1: 82.8125 (82.8573)  acc5: 97.7865 (97.6904)  time: 0.3811  data: 0.0001  max mem: 2753
Test:  [65/66]  eta: 0:00:00  loss: 0.5989 (0.6146)  acc1: 83.2031 (82.9140)  acc5: 97.5260 (97.6760)  time: 0.3655  data: 0.0001  max mem: 2753
Test: Total time: 0:00:42 (0.6448 s / it)
len of uncertainty_list: 50000
len of data_loader.dataset: 50000
* Acc@1 82.914 Acc@5 97.676 loss 0.615
Accuracy of the network on the 50000 train images: 82.9%
length of uncertainty_list: 50000
json dump into `outputs/trnData_uncertainty_list.json`
uncertainty_top100: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.001953125, 0.0029296875, 0.0029296875, 0.0029296875, 0.0029296875, 0.00360107421875, 0.00390625, 0.00390625, 0.00390625, 0.00390625, 0.00390625, 0.00390625, 0.00390625, 0.00390625, 0.00390625, 0.00390625, 0.00390625, 0.0048828125, 0.0048828125, 0.0048828125, 0.0048828125, 0.005859375, 0.005859375, 0.005859375, 0.005859375, 0.005859375, 0.005859375, 0.005859375, 0.00634765625, 0.0068359375, 0.0068359375, 0.0068359375, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0087890625, 0.0087890625, 0.009765625, 0.009765625, 0.009765625, 0.009765625, 0.009765625, 0.009765625, 0.009765625, 0.0107421875, 0.0107421875, 0.0107421875, 0.01123046875, 0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.0126953125, 0.0126953125, 0.0126953125, 0.0126953125, 0.0126953125, 0.0126953125, 0.0126953125, 0.0126953125, 0.013671875, 0.013671875, 0.013671875, 0.013671875, 0.0146484375, 0.0146484375, 0.0146484375, 0.0146484375, 0.0146484375, 0.0146484375, 0.015625, 0.015625]
cross_sample_ids: [75, 576, 782, 809, 848, 1062, 1601, 1634, 2537, 3012, 3047, 3063, 3171, 3232, 4101, 4589, 4835, 5002, 5019, 5074, 5241, 5437, 5693, 5774, 5982, 7556, 7784, 8169, 8189, 8226, 8403, 8953, 9077, 10533, 10690, 11837, 11853, 12666, 12873, 13485, 15125, 15533, 16025, 16747, 18850, 19635, 20007, 20228, 20592, 20767, 23291, 23542, 23953, 24202, 24277, 24421, 24673, 24703, 25625, 25857, 26667, 26792, 27324, 27399, 28838, 29246, 29403, 29916, 30979, 31054, 31072, 31350, 31677, 32745, 33788, 34254, 34580, 35071, 35501, 35760, 36402, 36652, 36709, 37076, 37176, 37900, 38062, 39018, 39869, 39883, 40125, 40238, 40407, 40707, 41752, 42316, 42779, 43129, 43177, 43468, 43661, 44787, 44903, 44906, 45203, 45430, 46488, 46546, 46567, 46760, 47117, 47223, 48121, 48181, 48366, 49294, 49701, 49732, 49822, 49964]
length of sample_ids_kmeans: 1000
length of cross sample ids: 120
json dump into `outputs/cross_sample_ids.json`

EOF
