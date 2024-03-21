import json
import os
import numpy as np
import random
import argparse
# import torch

import logging
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] ### %(message)s', level=logging.INFO)



def load_json(load_json_path):
    logging.info("[load_json] loading Start...")
    with open(load_json_path, 'r') as f:
        sample_ids = json.load(f)
    logging.info(f"[load_json] len of sample_ids: {len(sample_ids)}")
    logging.info("[load_json] loading Done...")
    return sample_ids


def json_dump(args, sample_ids):
    logging.info(f'[json_dump] start...')
    sample_num = len(sample_ids)
    logging.info(f'[json_dump] len of sample: {sample_num}')
    # if args.filename is None:
    #     name = args.feature_path.split("/")[-1]
    #     name = name[:-4]
    #     args.filename = name + "_comb_correct_sampleNum_%d.json" % (sample_num)
    output_path = os.path.join(args.output_dir, args.filename)
    logging.info(f'[json_dump] output dir: {output_path}')

    with open(output_path, "w") as f:
        json.dump(sample_ids, f)
    logging.info(f'[json_dump] done.')


def main(args):

    # load_json_path_1 = "features/CIFAR100_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_800.json"
    # load_json_path_2 = "/gpfsdata/home/wenshuai/projects/ActiveFT_xu/deit/outputs_cifar100/EMD800_s0/trnData_uncertainty_list.json"

    sample_ids = load_json(args.input_file1)
    uncertainty_val = load_json(args.input_file2)

    uncertainty_ids = sorted(range(len(uncertainty_val)), key=lambda k: uncertainty_val[k])

    assert len(sample_ids) < args.sample_num

    for e in uncertainty_ids:
        if e not in sample_ids:
            sample_ids.append(e)
        if len(sample_ids) == args.sample_num:
            break
    
    # std output
    json_dump(args, sample_ids)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    # parser.add_argument('--feature_path', default='features/CIFAR10_train.npy', type=str,help='path of saved features')
    parser.add_argument('--input_file1', default="features/CIFAR100_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_300_sampleNum_800.json", type=str, help='filename of the input json')
    parser.add_argument('--input_file2', default="/gpfsdata/home/wenshuai/projects/ActiveFT_xu/deit/outputs_cifar100/EMD800_s0/trnData_uncertainty_list.json", type=str, help='filename of the input json')
    parser.add_argument('--output_dir', default='features', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default="CIFAR100_train_EMD800_UN200_comb_sampleNum_1000.json", type=str, help='filename of the visualization')
    parser.add_argument('--sample_num', default=1000, type=int, help='sample number')
    args = parser.parse_args()
    main(args)

