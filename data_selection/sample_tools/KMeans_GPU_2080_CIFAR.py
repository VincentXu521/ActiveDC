import json
import os
import numpy as np
# import random
import argparse
# import functools
import torch
# import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
# from utils import *

import logging
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] ### %(message)s', level=logging.INFO)


# conda activate DAL_xu, faiss-gpu==1.7.2, for 2080 is ok, 3090 is NOT ok.
import faiss

class FaissKmeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   gpu=True)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        D, I = self.kmeans.index.search(X.astype(np.float32), 1)
        return D, I
        # return self.kmeans.index.search(X.astype(np.float32), 1)[1]


def cluster_features(features, sample_num):
    logging.info("[cluster_features] KMeans Start...")
    cluster_learner = FaissKmeans(n_clusters = sample_num)
    cluster_learner.fit(features)
    dists, indexs = cluster_learner.predict(features)
    logging.info("[cluster_features] KMeans Done!")
    return indexs, dists


def np2lst(indexs, dists):
    indexs = indexs.tolist()
    indexs = [e[0] for e in indexs] if isinstance(indexs[0], list) else indexs
    logging.info(f'[np2lst] indexs len: {len(indexs)}')                 # 50000
    logging.info(f'[np2lst] indexs set len: {len(set(indexs))}')        # 1000
    dists = dists.tolist()
    logging.info(f'[np2lst] before convert, dists[0]: {dists[0]}')
    dists = [e[0] for e in dists] if isinstance(dists[0], list) and len(dists[0])==1 else dists
    logging.info(f'[np2lst] after convert, dists[0]: {dists[0]}')
    return indexs, dists


def sample(indexs, dists, sample_num):
    logging.info("[sample] sample Start...")
    indexs, dists = np2lst(indexs, dists)
    sample_ids = {}
    minD = {}
    for j in range(len(indexs)):
        i, d = indexs[j], dists[j]
        if i not in minD or d < minD[i]:
            minD[i] = d
            sample_ids[i] = j
        elif d == minD[i]:
            logging.info(f'sample id {j} and {sample_ids[i]} distance({d}) equal')
    assert len(sample_ids) == sample_num, "len of sample_ids not equals to sample_num"
    sample_ids = list(sample_ids.values())
    sample_ids.sort()
    logging.info("[sample] sample Done...")
    return sample_ids


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
        args.filename = name + "_KMeans_2080_sampleNum_%d.json" % (sample_num)
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
    features = features / np.linalg.norm(features, 2, axis=1, keepdims=True)

    # cluster features via KMeans
    indexs, dists = cluster_features(features, sample_num)
    
    # sample ids nearest to centroids in features
    sample_ids = sample(indexs, dists, sample_num)

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


    """
    L2 Norm, way 1
    features = torch.Tensor(features).cuda()
    features = F.normalize(features, dim=1)
    features = features.detach().cpu().numpy()
    torch.cuda.empty_cache()
    
    L2 Norm, way 2
    from sklearn.preprocessing import normalize
    normalize(features, axis=1, norm='l2', copy=False)

    L2 Norm, way 3
    features = features / np.linalg.norm(features, 2, axis=1, keepdims=True)
    """
