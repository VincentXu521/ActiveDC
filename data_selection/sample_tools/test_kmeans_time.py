
# python sample_tools/test_kmeans_time.py -b CIFAR100 -d CIFAR100 --val --gpu -un -p 2

import os
import sys
import json
import argparse
# import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier as HistGBDT
from scipy.stats import wasserstein_distance

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


def Tukey_Transform(data, beta=0.5, info="data_name"):

    logging.info(f'[Tukey_Transform] [Start] {info}.shape: {data.shape}')
    if beta == 1:
        pass
    elif beta == 0:
        data = np.sign(data) * np.log(np.abs(data))
    else:
        data = np.sign(data) * np.power(np.abs(data), beta)
    logging.info(f'[Tukey_Transform] [Done] {info}.shape: {data.shape}')
    return data


def generate_base_data(base_features, base_labels):
    base_data = {}
    for e in set(base_labels):
        base_data[int(e)] = []
    for i in range(0, len(base_labels)):
        base_data[int(base_labels[i])].append(base_features[i])
    # logging.info(f"[generate_base_data] base_data.keys: {base_data.keys()}")
    return base_data


def base_mean_cov(args):
    base_features, base_labels = load_features(args.base_dataset)
    # base_features = Tukey_Transform(base_features, info="base_features")
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


def base_mean_cov_kmeans(args):
    base_features, base_labels = load_features(args.base_dataset)
    sample_num = len(set(base_labels))
    logging.info(f"[base_mean_cov_kmeans] KMeans Start...  Num_classes: {sample_num}")
    cluster_learner = FaissKmeans(n_clusters = sample_num)
    cluster_learner.fit(base_features)
    D, I = cluster_learner.predict(base_features)
    I = I.flatten()
    logging.info(f"[base_mean_cov_kmeans] KMeans fit and predict Done!")
    assert len(I) == len(base_features)
    assert len(set(I)) == sample_num

    base_data = {}
    for e in set(I):
        base_data[int(e)] = []
    for i in range(0, len(I)):
        base_data[int(I[i])].append(base_features[i])

    base_means = []
    base_cov = []
    for key in base_data.keys():
        feature = np.array(base_data[key])
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        base_means.append(mean)
        base_cov.append(cov)

    logging.info(f"[base_mean_cov_kmeans] Attention: base_data.keys() NOT real labels !!! ")
    return base_data, base_means, base_cov


def base_mean_cov_via_label(base_data, label):
    tmp = np.array(base_data[int(label)])
    mean = np.mean(tmp, axis=0)
    cov = np.cov(tmp.T)
    return mean, cov



def acc_via_classifier(args, trn_X, trn_y, val_X, val_y):
    
    logging.info(f"[{args.classifier}] -------------------------------------")
    logging.info(f"[{args.classifier}] classifier is building...")

    if args.classifier == "LR":
        classifier = LR(max_iter=1000).fit(X=trn_X, y=trn_y)
        # classifier = LR(max_iter=1000, C=0.5).fit(X=trn_X, y=trn_y)
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

    # index_flat = faiss.IndexFlatL2(dim)
    index_flat = faiss.IndexFlatIP(dim)  # cpu search 3 min 30s
    if args.gpu:
        # res = faiss.StandardGpuResources()
        # index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)  # one gpu search 3 min
        index_flat = faiss.index_cpu_to_all_gpus(index_flat)       # two gpu search 2 min 20s
        logging.info(f'[DC_aug] [GPU] faiss GPU index is used.')
    if not index_flat.is_trained:
        index_flat.train(features)
    logging.info(f'[DC_aug] index.is_trained = {index_flat.is_trained}')
    index_flat.add(features)
    logging.info(f'[DC_aug] index.ntotal = {index_flat.ntotal}\n')
    topK = 1  # search topK similarity

    sampled_data = []
    sampled_ids = []
    sampled_label = []
    # num_sampled = 5
    num_sampled = args.num_sampled
    sample_num = int(features.shape[0] * args.percent * 0.01)
    total_sampled = num_sampled * sample_num
    corrected_num = total_sampled
    fixed_num = 0

    logging.info(f'[DC_aug] === start Distribution Calibration and sample Aug === ')
    for i in range(sample_num):
        if args.log_aug:
            logging.info(f'[DC_aug] ================ {i} ================ ')
        # mean, cov = distribution_calibration(support_data[i], base_means, base_cov, k=args.knn, alpha=args.alpha)
        mean, cov = base_mean_cov_via_label(base_data, support_label[i])

        cnt_resample = 0

        while True:
            sample_virtual = np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled)
            sample_virtual = sample_virtual.astype(np.float32)
            faiss.normalize_L2(sample_virtual)

            D, I = index_flat.search(sample_virtual, topK)
            I = I.flatten()
            I = np.setdiff1d(I, support_ids)
            I = np.setdiff1d(I, sampled_ids)

            if(len(I) > 0):
                ids_old = np.concatenate([support_ids, sampled_ids], axis=0) if len(sampled_ids) > 0 else support_ids
                ids_old.sort()
                ids_new = np.concatenate([support_ids, sampled_ids, I]) if len(sampled_ids) > 0 else np.concatenate([support_ids, I])
                ids_new.sort()
                # ids_old = ids_old.astype(np.int64)
                # ids_new = ids_new.astype(np.int64)
                features_subset1 = features[ids_old]
                features_subset2 = features[ids_new]
                a_mean = features.mean(axis=1)
                b_mean = features_subset1.mean(axis=1)
                c_mean = features_subset2.mean(axis=1)
                emd_v1 = wasserstein_distance(a_mean, b_mean)
                emd_v2 = wasserstein_distance(a_mean, c_mean)
                if emd_v2 <= emd_v1:
                    break
                else:
                    cnt_resample += 1
                    logging.info(f'i = {i}: cnt_resample = {cnt_resample}, EMD distance worse, need resample with DC.')
            else:
                cnt_resample += 1
            if cnt_resample == 10:
                logging.info(f'[Timeout] resample more than 10 times.')
                break
        
        sample_reality = features[I]

        pseudo_label = [support_label[i]] * len(I)
        pseudo_index = 0
        for idx in I:
            # if cluster_classes[idx] != cluster_classes[support_ids[i]]:
            if labels[idx] != support_label[i]:
                if args.log_aug:
                    # logging.info(f"[DC_aug] ---------- {idx}th [{labels[idx]}] Error label [{support_label[i]}] ---------- ")
                    logging.info(f"[DC_aug] ---------- {idx}th [{labels[idx]}] Error label [{pseudo_label[pseudo_index]}] ---------- ")
                corrected_num -= 1
                # uncertainty
                if args.uncertainty:
                    mean_real = np.mean(np.array(base_data[int(labels[idx])]), axis=0)
                    mean_pred = np.mean(np.array(base_data[int(support_label[i])]), axis=0)
                    # mean_real, _ = base_mean_cov_via_label(base_data, labels[idx])
                    # mean_pred, _ = base_mean_cov_via_label(base_data, support_label[i])
                    distance1 = np.linalg.norm(features[idx] - mean_real)
                    distance2 = np.linalg.norm(features[idx] - mean_pred)
                    cos_sim_1 = features[idx].dot(mean_real) / (np.linalg.norm(features[idx]) * np.linalg.norm(mean_real))
                    cos_sim_2 = features[idx].dot(mean_pred) / (np.linalg.norm(features[idx]) * np.linalg.norm(mean_pred))
                    if distance1 <= distance2 or cos_sim_1 <= cos_sim_2:
                        if args.log_aug:
                            logging.info(f"[DC_aug] ++++++++++ {idx}th Error label will be fixed ++++++++++ ")
                        pseudo_label[pseudo_index] = labels[idx]
                        fixed_num += 1
                if idx in support_ids:
                    pseudo_label[pseudo_index] = labels[idx]
            pseudo_index += 1

        sampled_data.extend(sample_reality)
        sampled_ids.extend(I)
        # sampled_label.extend([support_label[i]] * num_sampled)
        sampled_label.extend(pseudo_label)

    logging.info(f'sampled_data len: {len(sampled_data)}')
    # sampled_data = Tukey_Transform(np.array(sampled_data), info="DC_sample_data")
    X_aug = np.concatenate([support_data, sampled_data])
    I_aug = np.concatenate([support_ids, sampled_ids])
    Y_aug = np.concatenate([support_label, sampled_label])

    logging.info(f"[DC_aug] sample corrected: {corrected_num} / {total_sampled} ({corrected_num / total_sampled * 100} %)")
    logging.info(f"[DC_aug] sample fixed num: {fixed_num}")
    logging.info(f"[DC_aug] sample corrected: {corrected_num + fixed_num} / {total_sampled} ({(corrected_num + fixed_num) / total_sampled * 100} %)")
    corrected_num_v6_2 = len(sampled_ids) - (total_sampled - corrected_num) + fixed_num
    total_sampled_v6_2 = len(sampled_ids)
    logging.info(f"[DC_aug] sample corrected: {corrected_num_v6_2} / {total_sampled_v6_2} ({corrected_num_v6_2 / total_sampled_v6_2 * 100} %) --- v6.2")
    return X_aug, Y_aug, I_aug


def resume_sample_ids(args, features, sample_num, erase_redundancy=True):
    if args.sample_resume == '':
        dataset_name = args.dataset
        features_num = 130000 if dataset_name == "ImageNet" else 50000
        sample_num = str(int(features_num * args.percent * 0.01))
        iter_str = "100" if dataset_name == "ImageNet" else "300"
        sample_resume = "DC_sample/" + dataset_name + "_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_" + iter_str + "_sampleNum_" + sample_num + ".json"
        logging.info(f'[resume_sample_ids] Auto combine path: {sample_resume}')
        if os.path.isfile(sample_resume):
            logging.info(f'[resume_sample_ids] args.sample_resume == '', and Auto match a file name ### fit !')
            # sample_ids = np.load(sample_resume)
            logging.info(f"[resume_sample_ids] resume sample ids from json file matched.")
            with open(sample_resume, 'r') as f:
                sample_ids = json.load(f)
            return sample_ids
        else:
            logging.info(f'[resume_sample_ids] args.sample_resume == '', and Auto match a file name ### miss.')

    if os.path.isfile(args.sample_resume) and args.sample_resume.endswith('npy'):
        logging.info(f"[resume_sample_ids] resume sample ids from npy file")
        sample_ids = np.load(args.sample_resume)
    elif os.path.isfile(args.sample_resume) and args.sample_resume.endswith('json'):
        logging.info(f"[resume_sample_ids] resume sample ids from json file")
        with open(args.sample_resume, 'r') as f:
            sample_ids = json.load(f)
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
    parser.add_argument('--beta', default=0.5, type=float, help='Tukey_Transform hyperparam beta')
    parser.add_argument('--knn', default=1, type=int, help='distribution_calibration hyperparam `k`')
    parser.add_argument('--alpha', default=0, type=float, help='distribution_calibration hyperparam alpha')
    parser.add_argument('-n', '--num_sampled', default=2, type=int, help='distribution_calibration sample number')
    parser.add_argument('--log_aug', action='store_true', default=False, help='log_aug_flag')
    parser.add_argument('--no_aug', action='store_true', default=False, help='Distribution Calibration Aug')
    parser.add_argument('--val', action='store_true', default=False, help='predict val dataset')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    parser.add_argument('-un', '--uncertainty', action='store_true', default=False, help='more labels via uncertainty')
    args = parser.parse_args()


    # ---- trn data loading
    dataset = args.dataset
    features, labels = load_features(dataset)  # include L2 norm
    # ---- val data loading
    val_features, val_label = load_features(dataset, dataset_type='val')

    num_features, num_classes, _ = dataset_info(labels)

    # ---- first feature sampling
    sample_num = int(features.shape[0] * args.percent * 0.01)
    sample_ids = resume_sample_ids(args, features, sample_num)
    # assert len(sample_ids) == sample_num, "len of sample_ids not equals to sample_num"
    # assert len(set(sample_ids)) == sample_num, "sample_ids redundancy"

    # v6.2: control DC second sample num
    threshold_per_cls = 50
    sample_num_per_cls = sample_num // num_classes
    args.num_sampled = 2 if sample_num_per_cls < threshold_per_cls else 1

    # ---- Tukey's transform
    # features = Tukey_Transform(features, info="trn_data")
    # val_data = Tukey_Transform(val_features, info="val_data")

    # ---- Base class statistics
    # base_data, base_means, base_cov = base_mean_cov(args)
    _, base_means, base_cov = base_mean_cov_kmeans(args)  # when have no `--un`(do NOT use base_data[real_label]), acc 80.38 same as above.

    sys.exit(1)

    # support data and query data
    support_ids = sample_ids
    support_data = features[support_ids]
    support_label = labels[support_ids]

    query_ids = [e for e in range(num_features) if e not in support_ids]
    query_data = features[query_ids]
    query_label = labels[query_ids]

    # ---- Tukey's transform
    support_data = Tukey_Transform(support_data, beta=args.beta, info="support_data")
    query_data = Tukey_Transform(query_data, beta=args.beta, info="query_data")
    val_data = Tukey_Transform(val_features, beta=args.beta, info="val_data")

    
    # ---- distribution calibration and second feature sampling
    X_aug, Y_aug, I_aug = DC_aug(args, features, labels, support_data, support_label, base_data, base_means, base_cov)
    # np.savez('features/' + args.dataset + '_sample_' + str(sample_num) + '_DC_indexs_labels', indexs=I_aug, labels=Y_aug)
    uncertainty = 'un_' if args.uncertainty else 'no_un_'
    np.savez('DC_sample/' + uncertainty + 'DC_' + args.dataset + '_EMD_sample_' + str(sample_num) + '_indexs_labels', indexs=I_aug, labels=Y_aug)

    # ---- train classifier
    trn_X, trn_y = (support_data, support_label) if args.no_aug else (X_aug, Y_aug)
    val_X, val_y = (val_data, val_label) if args.val else (query_data, query_label)
    
    test_data = "val" if args.val else "query"
    test_acc = acc_via_classifier(args, trn_X, trn_y, val_X, val_y)
    logging.info(f"[main] @Acc1 ({test_data}): {test_acc * 100} %")

    # from sklearn.preprocessing import KBinsDiscretizer
    # enc = KBinsDiscretizer(10, encode='onehot')
    # trn_bins = enc.fit_transform(trn_X, trn_y)
    # val_bins = enc.transform(val_X)
    # test_acc = acc_via_classifier(args, trn_bins, trn_y, val_bins, val_y)
    # logging.info(f"[main] [KBinsDiscretizer 10] @Acc1 ({test_data}): {test_acc * 100} %")



"""
python sample_tools/test_kmeans_time.py -b CIFAR100 -d CIFAR100 --val --gpu -un -p 2
---------------------
==============================
time cost KMeans: (10, 100, 1k)
---------------------------------------
cifar10:                40s
cifar100:              1min30s
imagenet-1k:       4min20s
---------------------------------------


==============================
time cost DC: (cifar100)
---------------------------------------
10%:     7min         5000
5%:       3min         2500
2%:       1.5min      1000
1%:       40s           500
---------------------------------------
"""

