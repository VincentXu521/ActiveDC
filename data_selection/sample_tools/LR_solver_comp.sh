#!/bin/sh
# cd data_selection && bash sample_tools/LR_solver_comp.sh 

# --solver: 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'
# --multi_class: 'auto', 'ovr', 'multinomial'

# nohup python sample_tools/LR_solver_comp.py -d CIFAR10 --val --no_aug -p 0.1 &> DC_sample/LR_solver_comp/CIFAR10_0.1.log &
# nohup python sample_tools/LR_solver_comp.py -d CIFAR10 --val --no_aug -p 0.1 -s lbfgs -m ovr &> DC_sample/LR_solver_comp/CIFAR10_0.1_lbfgs-ovr.log &
# nohup python sample_tools/LR_solver_comp.py -d CIFAR10 --val --no_aug -p 0.1 -s sag &> DC_sample/LR_solver_comp/CIFAR10_0.1_sag.log &
# nohup python sample_tools/LR_solver_comp.py -d CIFAR10 --val --no_aug -p 0.1 -s newton-cg &> DC_sample/LR_solver_comp/CIFAR10_0.1_newton-cg.log &
# nohup python sample_tools/LR_solver_comp.py -d CIFAR10 --val --no_aug -p 0.1 -s newton-cholesky &> DC_sample/LR_solver_comp/CIFAR10_0.1_newton-cholesky.log &
# nohup python sample_tools/LR_solver_comp.py -d CIFAR10 --val --no_aug -p 0.1 -c Perceptron &> DC_sample/LR_solver_comp/CIFAR10_0.1_Perceptron.log &

per=5
dataset=ImageNet

echo "LR start: " $dataset $per
date

python sample_tools/LR_solver_comp.py -d $dataset --val --no_aug -p $per &> DC_sample/LR_solver_comp/${dataset}_${per}.log
wait; echo "default: lbfgs && MvM"
tail -2 DC_sample/LR_solver_comp/${dataset}_${per}.log
echo "----------------------------------"
python sample_tools/LR_solver_comp.py -d $dataset --val --no_aug -p $per -s lbfgs -m ovr &> DC_sample/LR_solver_comp/${dataset}_${per}_lbfgs-ovr.log
wait; echo "lbfgs && OvR"
tail -2 DC_sample/LR_solver_comp/${dataset}_${per}_lbfgs-ovr.log
echo "----------------------------------"
python sample_tools/LR_solver_comp.py -d $dataset --val --no_aug -p $per -s sag &> DC_sample/LR_solver_comp/${dataset}_${per}_sag.log
wait; echo "sag && MvM"
tail -2 DC_sample/LR_solver_comp/${dataset}_${per}_sag.log
echo "----------------------------------"
python sample_tools/LR_solver_comp.py -d $dataset --val --no_aug -p $per -s newton-cg &> DC_sample/LR_solver_comp/${dataset}_${per}_newton-cg.log
wait; echo "newton-cg && MvM"
tail -2 DC_sample/LR_solver_comp/${dataset}_${per}_newton-cg.log
echo "----------------------------------"
python sample_tools/LR_solver_comp.py -d $dataset --val --no_aug -p $per -s newton-cholesky &> DC_sample/LR_solver_comp/${dataset}_${per}_newton-cholesky.log
wait; echo "newton-cholesky && OvR"
tail -2 DC_sample/LR_solver_comp/${dataset}_${per}_newton-cholesky.log
echo "----------------------------------"
python sample_tools/LR_solver_comp.py -d $dataset --val --no_aug -p $per -c Perceptron &> DC_sample/LR_solver_comp/${dataset}_${per}_Perceptron.log
wait; echo "Perceptron && MvM"
tail -2 DC_sample/LR_solver_comp/${dataset}_${per}_Perceptron.log
echo "----------------------------------"
python sample_tools/LR_solver_comp.py -d $dataset --val --no_aug -p $per -c MLP &> DC_sample/LR_solver_comp/${dataset}_${per}_MLP.log
wait; echo "MLP && MvM"
tail -2 DC_sample/LR_solver_comp/${dataset}_${per}_MLP.log
echo "=================================="


echo "LR end"
echo "log save path: data_selection/DC_sample/LR_solver_comp"
date
