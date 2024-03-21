# cd deit/

python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 \
        --use_env main_eval.py \
        --eval \
        --dist-eval \
        --data-set CIFAR10 \
        --resume outputs_combination_correct/best_checkpoint.pth \
        --output_dir outputs_combination_correct


<< EOF
----------------------------------
dataset_train, --nproc_per_node=1
----------------------------------

(active_finetune) wenshuai@node21:~/projects/ActiveFT_xu/deit$ bash eval_trnData.sh 
...
Test:  [ 0/66]  eta: 0:07:35  loss: 0.6693 (0.6693)  acc1: 82.6823 (82.6823)  acc5: 97.5260 (97.5260)  time: 6.9066  data: 5.9328  max mem: 2753
Test:  [10/66]  eta: 0:01:26  loss: 0.6854 (0.6908)  acc1: 80.7292 (80.7173)  acc5: 97.3958 (97.3248)  time: 1.5452  data: 0.9805  max mem: 2753
Test:  [20/66]  eta: 0:00:47  loss: 0.6799 (0.6813)  acc1: 80.8594 (80.9524)  acc5: 97.5260 (97.4702)  time: 0.7403  data: 0.2429  max mem: 2753
Test:  [30/66]  eta: 0:00:30  loss: 0.6757 (0.6783)  acc1: 81.1198 (81.1240)  acc5: 97.3958 (97.3412)  time: 0.4753  data: 0.0005  max mem: 2753
Test:  [40/66]  eta: 0:00:19  loss: 0.6663 (0.6735)  acc1: 81.6406 (81.3040)  acc5: 97.2656 (97.3736)  time: 0.4740  data: 0.0004  max mem: 2753
Test:  [50/66]  eta: 0:00:10  loss: 0.6660 (0.6718)  acc1: 81.6406 (81.3777)  acc5: 97.1354 (97.3167)  time: 0.4271  data: 0.0002  max mem: 2753
Test:  [60/66]  eta: 0:00:03  loss: 0.6784 (0.6740)  acc1: 80.8594 (81.3098)  acc5: 97.2656 (97.3467)  time: 0.3837  data: 0.0001  max mem: 2753
Test:  [65/66]  eta: 0:00:00  loss: 0.6784 (0.6740)  acc1: 80.4688 (81.3100)  acc5: 97.2656 (97.3540)  time: 0.3667  data: 0.0001  max mem: 2753
Test: Total time: 0:00:40 (0.6194 s / it)
len of uncertainty_list: 50000
len of data_loader.dataset: 50000
* Acc@1 81.310 Acc@5 97.354 loss 0.674
Accuracy of the network on the 50000 train images: 81.3%
length of uncertainty_list: 50000
json dump into `outputs_combination_correct/trnData_uncertainty_list.json`

EOF
