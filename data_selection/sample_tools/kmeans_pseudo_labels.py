import json
import os
import numpy as np

import argparse
import torch
import torch.nn.functional as F

import logging
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] ### %(message)s', level=logging.INFO)

# conda activate active_finetune (3090, faiss-gpu==1.7.4)
# export LD_LIBRARY_PATH=/gpfsdata/home/wenshuai/miniconda3/envs/active_finetune/lib:$LD_LIBRARY_PATH
import faiss

class FaissKmeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, use_gpu=True):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.use_gpu = use_gpu
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   gpu=self.use_gpu)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]
        # D, I = self.kmeans.index.search(X.astype(np.float32), 1)
        # return D, I


def cluster_features(features, label_num):
    logging.info("[cluster_features] KMeans Start...")
    cluster_learner = FaissKmeans(n_clusters = label_num)
    # cluster_learner = FaissKmeans(n_clusters = label_num, max_iter=1000)
    cluster_learner.fit(features)
    indexs = cluster_learner.predict(features)
    logging.info("[cluster_features] KMeans Done!")
    return indexs


def np2lst(indexs):
    indexs = indexs.tolist()
    indexs = [e[0] for e in indexs] if isinstance(indexs[0], list) else indexs
    logging.info(f'[np2lst] indexs len: {len(indexs)}')                 # 50000
    logging.info(f'[np2lst] indexs set len: {len(set(indexs))}')        # 10
    return indexs


def load_features(args):
    logging.info("[load_features] loading Start...")
    input = np.load(args.feature_path)
    # features, _ = input[:, :-1], input[:, -1]
    features, labels = input[:, :-1], input[:, -1]
    logging.info("[load_features] loading Done...")
    logging.info(f'[load_features] features.shape: {features.shape}')      # features.shape: (50000, 384)
    logging.info(f'[load_features] labels.shape: {labels.shape}')          # labels.shape: (50000,)
    return features, labels


def json_dump(args, pseudo_labels):
    logging.info(f'[json_dump] start...')
    if args.filename is None:
        name = args.feature_path.split("/")[-1]
        name = name[:-4]
        args.filename = name + "_pseudo_labels.json"
    output_path = os.path.join(args.output_dir, args.filename)
    logging.info(f'[json_dump] output dir: {output_path}')

    with open(output_path, "w") as f:
        json.dump(pseudo_labels, f)
    logging.info(f'[json_dump] done.')



def main(args):

    if args.resume:
        with open(args.resume, 'r') as f:
            pseudo_labels = json.load(f)
        logging.info(f'[Resume] len of pseudo_labels: {len(pseudo_labels)}')
        return

    # load_features
    features, labels = load_features(args)

    # features L2 Norm
    features = features / np.linalg.norm(features, 2, axis=1, keepdims=True)

    # cluster features via KMeans
    label_num = len(set(labels.tolist()))
    logging.info(f'label_num: {label_num}')
    pseudo_labels = cluster_features(features, label_num)

    logging.info(f'pseudo_labels.shape: {pseudo_labels.shape}')
    pseudo_labels = np2lst(pseudo_labels)

    assert len(pseudo_labels) == len(labels)

    pseudo_labels_dict = {}
    trueth_labels_dict = {}
    for i in range(label_num):
        pseudo_labels_dict[i] = set()
        trueth_labels_dict[i] = set()

    for i in range(len(labels)):
        pseudo_labels_dict[pseudo_labels[i]].add(i)
        trueth_labels_dict[labels[i]].add(i)

    pseudo_labels_dict_new = {}
    for i in range(label_num):
        maxLen = 0
        maxSim = -1
        for j in range(label_num):
            curLen = len(pseudo_labels_dict[i] & trueth_labels_dict[j])
            if curLen > maxLen:
                maxLen = curLen
                maxSim = j
        pseudo_labels_dict_new[maxSim] = pseudo_labels_dict[i]


    logging.info(f'len of pseudo_labels_dict_new: {len(pseudo_labels_dict_new)}')
    logging.info(f'len of trueth_labels_dict: {len(trueth_labels_dict)}')
    assert len(pseudo_labels_dict) == len(trueth_labels_dict)
    assert len(pseudo_labels_dict_new) == len(trueth_labels_dict)
    # assert sorted(list(pseudo_labels_dict_new.values())) == sorted(list(trueth_labels_dict.values()))
    # assert sorted(list(pseudo_labels_dict_new.values())) == sorted(list(range(len(labels))))

    pseudo_labels_new = [-1] * len(labels)
    for k in pseudo_labels_dict_new.keys():
        for e in pseudo_labels_dict_new[k]:
            pseudo_labels_new[e] = k

    
    assert all([e+1 for e in pseudo_labels_new])
    # assert sorted(pseudo_labels_new) == sorted(pseudo_labels)
    cnt = 0
    for i in range(len(labels)):
        if pseudo_labels_new[i] == labels[i]:
            cnt += 1
    acc = cnt / len(labels)
    logging.info(f'pseudo labels @Acc: {acc}')
    logging.info(f'pseudo labels @Acc: {acc * 100} %')  ### pseudo labels @Acc: 79.642 %

    # std output
    # logging.info(pseudo_labels_new)
    # json_dump(args, pseudo_labels_new)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    parser.add_argument('-f', '--feature_path', default='features/CIFAR10_train.npy', type=str,help='path of saved features')
    parser.add_argument('--output_dir', default='features', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default=None, type=str, help='filename of the visualization')
    parser.add_argument('-r', '--resume', default='', type=str, help='path of pseudo_labels_new json file')
    args = parser.parse_args()
    main(args)

