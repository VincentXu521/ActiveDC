import json
import os
import numpy as np
import random
import argparse
# import torch

import logging
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] ### %(message)s', level=logging.INFO)


def load_features(args):
    logging.info("[load_features] loading Start...")
    input = np.load(args.feature_path)
    # features, _ = input[:, :-1], input[:, -1]
    features, labels = input[:, :-1], input[:, -1]
    logging.info("[load_features] loading Done...")
    logging.info(f'[load_features] features.shape: {features.shape}')      # features.shape: (50000, 384)
    logging.info(f'[load_features] labels.shape: {labels.shape}')          # labels.shape: (50000,)
    return features, labels


def json_dump(args, sample_ids):
    logging.info(f'[json_dump] start...')
    sample_num = len(sample_ids)
    logging.info(f'[json_dump] len of sample: {sample_num}')
    if args.filename is None:
        name = args.feature_path.split("/")[-1]
        name = name[:-4]
        args.filename = name + "_random_sampleNum_%d.json" % (sample_num)
    output_path = os.path.join(args.output_dir, args.filename)
    logging.info(f'[json_dump] output dir: {output_path}')

    with open(output_path, "w") as f:
        json.dump(sample_ids, f)
    logging.info(f'[json_dump] done.')


def main(args):

    # load_features
    features, labels = load_features(args)

    total_num = features.shape[0]
    sample_num = int(total_num * args.percent * 0.01)

    # features L2 Norm
    # features = features / np.linalg.norm(features, 2, axis=1, keepdims=True)
    
    # random sample
    logging.info(f'[random sample] start...')
    sample_ids = random.sample(range(total_num), sample_num)
    logging.info(f'[random sample] done.')

    # std output
    logging.info(sample_ids)
    json_dump(args, sample_ids)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    parser.add_argument('--feature_path', default='features/CIFAR10_train.npy', type=str,help='path of saved features')
    parser.add_argument('--output_dir', default='features', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default=None, type=str, help='filename of the visualization')
    parser.add_argument('--percent', default=2, type=float, help='sample percent')
    args = parser.parse_args()
    main(args)



