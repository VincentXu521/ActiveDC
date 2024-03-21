import sys
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier as HistGBDT

import logging
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] ### %(message)s', level=logging.INFO)

# conda activate active_finetune (3090, faiss-gpu==1.7.4)
# export LD_LIBRARY_PATH=/gpfsdata/home/wenshuai/miniconda3/envs/active_finetune/lib:$LD_LIBRARY_PATH
import faiss



# a hyper-parameter: alpha
def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    # mean = np.array(base_means)[index]
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0) + alpha

    return calibrated_mean, calibrated_cov



def load_features(dataset, dataset_type='train'):
    assert dataset_type == 'train' or 'val', "dataset_type must be `train` or `val`"

    if dataset == "CIFAR10":
        feature_path = "features/CIFAR10_" + dataset_type + ".npy"
    elif dataset == "CIFAR100":
        feature_path = "features/CIFAR100_" + dataset_type + ".npy"
    elif dataset == "ImageNet":
        feature_path = "features/ImageNet_" + dataset_type + ".npy"
    else:
        logging.info("[load_features] unknowed dataset name ...")
        return None, None

    logging.info(f"[load_features] [{dataset_type}] loading Start...")
    input = np.load(feature_path)
    features, labels = input[:, :-1], input[:, -1]
    # features L2 Norm
    # features = features / np.linalg.norm(features, 2, axis=1, keepdims=True)
    features = features.astype(np.float32)
    faiss.normalize_L2(features)
    logging.info(f"[load_features] [{dataset_type}] loading Done...")
    logging.info(f'[load_features] [{dataset_type}] features.shape: {features.shape}')      # features.shape: (-, 384)
    logging.info(f'[load_features] [{dataset_type}] labels.shape: {labels.shape}')          # labels.shape: (-,)
    return features, labels


def dataset_info(labels, name='train_dataset'):
    num_features = len(labels)
    num_classes  = len(set(labels))
    num_per_class= num_features // num_classes
    logging.info(f"[dataset_info] [{name}] number of features: {num_features}")
    logging.info(f"[dataset_info] [{name}] number of classes : {num_classes}")
    logging.info(f"[dataset_info] [{name}] number per class  : {num_per_class}")
    return num_features, num_classes, num_per_class


def generate_base_data(base_features, base_labels):
    base_data = {}
    for e in set(base_labels):
        base_data[int(e)] = []
    for i in range(0, len(base_labels)):
        base_data[int(base_labels[i])].append(base_features[i])
    return base_data


def base_mean_cov(args):
    base_features, base_labels = load_features(args.base_dataset)
    base_data = generate_base_data(base_features, base_labels)
    base_means = []
    base_cov = []
    for key in base_data.keys():
        feature = np.array(base_data[key])
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        base_means.append(mean)
        base_cov.append(cov)
    return base_data, base_means, base_cov


def Tukey_Transform(data, beta=0.5, info="data_name"):
    if beta == 0:
        data = np.sign(data) * np.log(np.abs(data))
    else:
        data = np.sign(data) * np.power(np.abs(data), beta)

    logging.info(f'[Tukey_Transform] {info}.shape: {data.shape}')
    return data


def acc_via_classifier(args, trn_X, trn_y, val_X, val_y):
    
    logging.info(f"[{args.classifier}] -------------------------------------")
    logging.info(f"[{args.classifier}] classifier is building...")

    if args.classifier == "LR":
        # classifier = LR(max_iter=1000).fit(X=trn_X, y=trn_y)
        classifier = LR(max_iter=1000, C=0.5).fit(X=trn_X, y=trn_y)
        # classifier = LogisticRegressionCV(max_iter=1000, cv=5).fit(X=trn_X, y=trn_y)
    elif args.classifier == "SVM":
        # classifier = svm.SVC(kernel='linear', decision_function_shape='ovo').fit(X=trn_X, y=trn_y)  # more time: N*(n-1)/2
        classifier = svm.SVC(kernel='linear').fit(X=trn_X, y=trn_y)  # default='ovr', less time: N
        # classifier = svm.LinearSVC(multi_class='crammer_singer').fit(X=trn_X, y=trn_y)
    elif args.classifier == "GBDT":
        classifier = HistGBDT(max_iter=1000).fit(X=trn_X, y=trn_y)
    else:
        logging.info(f"[acc_via_classifier] args.classifer: {args.classifer} is unknown.")
        return -1

    if args.no_aug:
        logging.info(f"[{args.classifier}] classifier w.o. DC Aug")
    else:
        logging.info(f"[{args.classifier}] classifier with DC Aug")

    if args.val:
        logging.info(f"[{args.classifier}] predict `val` dataset")
    else:
        logging.info(f"[{args.classifier}] predict `query` dataset")

    logging.info(f"[{args.classifier}] classifier predict start...")
    predicts = classifier.predict(val_X)
    logging.info(f"[{args.classifier}] classifier predict done.")

    acc_score = accuracy_score(val_y, predicts)
    logging.info(f"[{args.classifier}] @Acc1 : {acc_score}")
    return acc_score



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


def cluster_and_sample(features, sample_num):
    logging.info("[cluster_and_sample] KMeans Start...")
    cluster_learner = FaissKmeans(n_clusters = sample_num)
    cluster_learner.fit(features)
    # dists, indexs = cluster_learner.predict(features)
    logging.info("[cluster_and_sample] KMeans Done!")

    logging.info("[cluster_and_sample] sample Start...")
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    D, I = index.search(cluster_learner.cluster_centers_, 1)  # find the 1 nearest points in `features` to `cluster_centers_`.
    sample_ids = sorted(I.flatten())
    logging.info("[cluster_and_sample] sample Done!")
    return sample_ids


def DC_aug(args, features, labels, support_data, support_label, base_data, base_means, base_cov):
    if args.no_aug:
        logging.info(f'[DC_aug] do NoT apply distribution calibration aug.')
        return None, None, None

    logging.info(f'[DC_aug] uncertainty AL is included: {args.uncertainty}')
    dim = features.shape[1]  # 384
    logging.info(f'[DC_aug] dim == {dim}')
    # index_factory = faiss.IndexFlatL2(dim)
    index_factory = faiss.IndexFlatIP(dim)
    if not index_factory.is_trained:
        index_factory.train(features)
    logging.info(f'[DC_aug] index.is_trained = {index_factory.is_trained}')
    index_factory.add(features)
    logging.info(f'[DC_aug] index.ntotal = {index_factory.ntotal}\n')
    topK = 1  # search topK similarity

    sampled_data = []
    sampled_ids = []
    sampled_label = []
    # num_sampled = 5
    num_sampled = 2
    sample_num = int(features.shape[0] * args.percent * 0.01)
    total_sampled = num_sampled * sample_num
    corrected_num = total_sampled
    fixed_num = 0

    logging.info(f'[DC_aug] === start Distribution Calibration and sample Aug === ')
    for i in range(sample_num):
        if args.log_aug:
            logging.info(f'[DC_aug] ================ {i} ================ ')
        mean, cov = distribution_calibration(support_data[i], base_means, base_cov, k=1, alpha=0)
        tmp = np.array(base_data[int(support_label[i])])
        mean = np.mean(tmp, axis=0)
        cov = np.cov(tmp.T)
        # logging.info(f'[main] mean: {mean}')
        # logging.info(f'[main] tmp_mean: {tmp_mean}')
        # logging.info(f'[main] mean == tmp_mean: {all(mean==tmp_mean)}')
        sample_virtual = np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled)
        sample_virtual = sample_virtual.astype(np.float32)
        faiss.normalize_L2(sample_virtual)

        D, I = index_factory.search(sample_virtual, topK)
        I = I.flatten()
        sample_reality = features[I]

        pseudo_label = [support_label[i]] * num_sampled * topK
        pseudo_index = 0
        for idx in I:
            if labels[idx] != support_label[i]:
                if args.log_aug:
                    # logging.info(f"[DC_aug] ---------- {idx}th [{labels[idx]}] Error label [{support_label[i]}] ---------- ")
                    logging.info(f"[DC_aug] ---------- {idx}th [{labels[idx]}] Error label [{pseudo_label[pseudo_index]}] ---------- ")
                corrected_num -= 1
                # uncertainty
                mean_real = np.mean(np.array(base_data[int(labels[idx])]), axis=0)
                mean_pred = np.mean(np.array(base_data[int(support_label[i])]), axis=0)
                distance1 = np.linalg.norm(features[idx] - mean_real)
                distance2 = np.linalg.norm(features[idx] - mean_pred)
                if args.uncertainty and distance1 <= distance2:
                    if args.log_aug:
                        logging.info(f"[DC_aug] ++++++++++ {idx}th Error label will be fixed ++++++++++ ")
                    pseudo_label[pseudo_index] = labels[idx]
                    fixed_num += 1
            pseudo_index += 1

        sampled_data.extend(sample_reality)
        sampled_ids.extend(I)
        # sampled_label.extend([support_label[i]] * num_sampled)
        sampled_label.extend(pseudo_label)

    X_aug = np.concatenate([support_data, sampled_data])
    I_aug = np.concatenate([support_ids, sampled_ids])
    Y_aug = np.concatenate([support_label, sampled_label])

    logging.info(f"[DC_aug] sample corrected: {corrected_num} / {total_sampled} ({corrected_num / total_sampled * 100} %)")
    logging.info(f"[DC_aug] sample fixed num: {fixed_num}")
    return X_aug, Y_aug, I_aug


def resume_sample_ids(args, features, sample_num, erase_redundancy=True):
    if args.sample_resume:
        logging.info(f"[resume_sample_ids] resume sample ids")
        sample_ids = np.load(args.sample_resume)
    else:
        # cluster features via KMeans and sample ids nearest to centroids in features
        if erase_redundancy:
            from distribution_calibration import cluster_features, sample
            indexs, dists = cluster_features(features, sample_num)
            sample_ids = sample(indexs, dists, sample_num)
        else:
            sample_ids = cluster_and_sample(features, sample_num)
        
        save_path = "kmeans_sample_ids/" + args.dataset + "_" + str(sample_num) + "_sample_ids.npy"
        np.save(save_path, sample_ids)
    return sample_ids




if __name__ == '__main__':

    parser = argparse.ArgumentParser('distribution calibration')
    parser.add_argument('-b', '--base_dataset', default='ImageNet', choices=['CIFAR10', 'CIFAR100', 'ImageNet'],type=str, help='base dataset name')
    parser.add_argument('-d', '--dataset', default='ImageNet', choices=['CIFAR10', 'CIFAR100', 'ImageNet'],type=str, help='dataset name')
    parser.add_argument('-p', '--percent', default=2, type=float, help='sample percent')
    parser.add_argument('-c', '--classifier', default='LR', choices=['LR', 'SVM', 'GBDT'],type=str, help='classifer name')
    parser.add_argument('-r', '--sample_resume', default='', type=str, help='first sample(kmeans) ids file path')
    parser.add_argument('--log_aug', action='store_true', default=False, help='log_aug_flag')
    parser.add_argument('--no_aug', action='store_true', default=False, help='Distribution Calibration Aug')
    parser.add_argument('--val', action='store_true', default=False, help='predict val dataset')
    parser.add_argument('-un', '--uncertainty', action='store_true', default=False, help='more labels via uncertainty')
    args = parser.parse_args()


    # ---- trn data loading
    dataset = args.dataset
    features, labels = load_features(dataset)  # include L2 norm
    # ---- val data loading
    val_features, val_label = load_features(dataset, dataset_type='val')

    num_features, _, _ = dataset_info(labels)

    # ---- first feature sampling
    sample_num = int(features.shape[0] * args.percent * 0.01)
    sample_ids = resume_sample_ids(args, features, sample_num)
    assert len(sample_ids) == sample_num, "len of sample_ids not equals to sample_num"
    # assert len(set(sample_ids)) == sample_num, "sample_ids redundancy"
    
    # ---- Base class statistics
    base_data, base_means, base_cov = base_mean_cov(args)

    # support data and query data
    support_ids = sample_ids
    support_data = features[support_ids]
    support_label = labels[support_ids]

    query_ids = [e for e in range(num_features) if e not in support_ids]
    query_data = features[query_ids]
    query_label = labels[query_ids]

    # ---- Tukey's transform
    support_data = Tukey_Transform(support_data, info="support_data")
    query_data = Tukey_Transform(query_data, info="query_data")
    val_data = Tukey_Transform(val_features, info="val_data")
    
    # ---- distribution calibration and second feature sampling
    X_aug, Y_aug, I_aug = DC_aug(args, features, labels, support_data, support_label, base_data, base_means, base_cov)

    # ---- train classifier
    trn_X, trn_y = (support_data, support_label) if args.no_aug else (X_aug, Y_aug)
    val_X, val_y = (val_data, val_label) if args.val else (query_data, query_label)
    
    test_data = "val" if args.val else "query"
    test_acc = acc_via_classifier(args, trn_X, trn_y, val_X, val_y)
    logging.info(f"[main] @Acc1 ({test_data}): {test_acc * 100} %")




"""
python sample_tools/DC_v1.py -p 1 --un
---------------------
1300 + __ --> 84.64%
1300 + __ --> 84.52%
1300 + 98 --> 84.58%
========================================================================
python sample_tools/DC_v1.py -p 1 --val --un
---------------------
1300 + 83 --> 81.44%
1300 + __ --> 81.46%
1300 + 90 --> 81.06%
1300 +102 --> 81.80%
1300 +116 --> 81.38%
1300 +121 --> 81.46%
1300 +104 --> 81.34%
========================================================================
python sample_tools/DC_v1.py -p 1 --val --no_aug
---------------------
79.70 * 6, beta = 0.5;
79.78 * 3, beta = 0.5; ( LR, when `erase_redundancy` is True )
80.00 * 3, beta = 0.5; ( LR, when `erase_redundancy` is True, C=0.6 or 0.5 )
79.76 * 1, beta = 0.5; ( LR, C=1.35 )
79.50 * 5, beta = 0.5; ( LRCV, cv=5 )
76.40 * 2, beta = 0.5; ( SVM 'ovr' && max_iter=1000 or not)
77.98 * 1, beta = 0.5; ( LinearSVC )
78.38 * 1, beta = 0.5; ( LinearSVC && multi_class='crammer_singer')
78.94 * 4, beta = 0.5; ( SVM ('ovr' or 'ovo') && kernel='linear')
69.28 * 3, beta = 0.5; ( SVM && kernel='poly')
63.14 * 1, beta = 0.5; ( GBDT )
77.04 * 2, beta = 1.0;
========================================================================
python sample_tools/DC_v1.py -p 1 --val
---------------------
1300 + 0 --> 80.74
"""
