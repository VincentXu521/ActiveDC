# cd deit/

python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 \
        --use_env main_eval.py \
        --eval \
        --dist-eval \
        --data-set CIFAR100 \
        --resume outputs_cifar100/EMD800_s0/best_checkpoint.pth \
        --output_dir outputs_cifar100/EMD800_s0


<< EOF
----------------------------------
dataset_train, --nproc_per_node=1
----------------------------------
(active_finetune) wenshuai@node21:~/projects/ActiveFT_xu/deit$ bash eval_trnData_cifar100.sh 
...
Files already downloaded and verified
Files already downloaded and verified
Creating model: deit_small_patch16_224
number of params: 21704164
Test:  [ 0/66]  eta: 0:11:33  loss: 3.4651 (3.4651)  acc1: 25.7812 (25.7812)  acc5: 51.1719 (51.1719)  time: 10.5127  data: 9.3885  max mem: 2754
Test:  [10/66]  eta: 0:01:22  loss: 3.5003 (3.5191)  acc1: 25.6510 (25.6747)  acc5: 48.1771 (48.6032)  time: 1.4812  data: 0.9038  max mem: 2754
Test:  [20/66]  eta: 0:00:46  loss: 3.4990 (3.5068)  acc1: 25.6510 (25.9115)  acc5: 48.1771 (48.6793)  time: 0.5345  data: 0.0278  max mem: 2754
Test:  [30/66]  eta: 0:00:30  loss: 3.4878 (3.5077)  acc1: 25.6510 (25.7014)  acc5: 48.6979 (48.6643)  time: 0.4869  data: 0.0003  max mem: 2754
Test:  [40/66]  eta: 0:00:19  loss: 3.4966 (3.5045)  acc1: 25.6510 (25.7971)  acc5: 48.8281 (48.7805)  time: 0.4884  data: 0.0074  max mem: 2754
Test:  [50/66]  eta: 0:00:11  loss: 3.4998 (3.5038)  acc1: 25.9115 (25.8323)  acc5: 48.8281 (48.9941)  time: 0.4524  data: 0.0073  max mem: 2754
Test:  [60/66]  eta: 0:00:03  loss: 3.5022 (3.5011)  acc1: 25.7812 (25.8581)  acc5: 48.6979 (49.0288)  time: 0.3996  data: 0.0001  max mem: 2754
Test:  [65/66]  eta: 0:00:00  loss: 3.5022 (3.5010)  acc1: 25.9115 (25.8800)  acc5: 48.7500 (49.0120)  time: 0.3726  data: 0.0001  max mem: 2754
Test: Total time: 0:00:40 (0.6211 s / it)
len of uncertainty_list: 50000
len of data_loader.dataset: 50000
* Acc@1 25.880 Acc@5 49.012 loss 3.501
Accuracy of the network on the 50000 train images: 25.9%
length of uncertainty_list: 50000
json dump into `outputs_cifar100/EMD800_s0/trnData_uncertainty_list.json`
----------------------------------
EOF
