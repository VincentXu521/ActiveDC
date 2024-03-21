import sys
import argparse
import numpy as np

import logging
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] ### %(message)s', level=logging.INFO)



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
    logging.info(f"[load_features] [{dataset_type}] loading Done...")
    logging.info(f'[load_features] [{dataset_type}] features.shape: {features.shape}')      # features.shape: (-, 384)
    logging.info(f'[load_features] [{dataset_type}] labels.shape: {labels.shape}')          # labels.shape: (-,)
    return features, labels


def dataset_info(labels):
    num_features = len(labels)
    num_classes  = len(set(labels))
    num_per_class= num_features // num_classes
    logging.info(f"[dataset_info] number of features: {num_features}")
    logging.info(f"[dataset_info] number of classes : {num_classes}")
    logging.info(f"[dataset_info] number per class  : {num_per_class}")
    return num_features, num_classes, num_per_class


def cifar_is_unsorted(args, labels):
    from torchvision import datasets
    if args.dataset == "CIFAR10":
        dataset = datasets.CIFAR10(root="data", train = True, download=False, transform=None)
    elif args.dataset == "CIFAR100":
        dataset = datasets.CIFAR100(root="data", train = True, download=False, transform=None)
    else:
        raise NotImplementedError
    return labels[-3] == dataset[-3][1] and labels[-2] == dataset[-2][1] and labels[-1] == dataset[-1][1]


def feature_is_sorted(labels):
    # print(all(labels == sorted(labels)))  # True or False
    return all([labels[i] <= labels[i + 1] for i in range(len(labels) - 1)])


def generate_base_data(base_features, base_labels):
    base_data = {}
    for e in set(labels):
        base_data[int(e)] = []
    for i in range(0, len(labels)):
        base_data[int(labels[i])].append(base_features[i])
    return base_data


# conda activate active_finetune (3090, faiss-gpu==1.7.4)
# export LD_LIBRARY_PATH=/gpfsdata/home/wenshuai/miniconda3/envs/active_finetune/lib:$LD_LIBRARY_PATH
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



if __name__ == '__main__':

    parser = argparse.ArgumentParser('distribution calibration')
    parser.add_argument('-b', '--base_dataset', default='ImageNet', choices=['CIFAR10', 'CIFAR100', 'ImageNet'],type=str, help='base dataset name')
    parser.add_argument('-d', '--dataset', default='ImageNet', choices=['CIFAR10', 'CIFAR100', 'ImageNet'],type=str, help='dataset name')
    parser.add_argument('-p', '--percent', default=2, type=float, help='sample percent')
    parser.add_argument('--log_aug', action='store_true', default=False, help='log_aug_flag')
    parser.add_argument('--no_aug', action='store_true', default=False, help='Distribution Calibration Aug')
    parser.add_argument('--val', action='store_true', default=False, help='predict val dataset')
    parser.add_argument('--uncertainty', action='store_true', default=False, help='more labels via uncertainty')
    args = parser.parse_args()


    # ---- data loading
    dataset = args.dataset  # CIFAR10, CIFAR100, ImageNet
    features, labels = load_features(dataset)

    num_features, num_classes, num_per_class = dataset_info(labels)

    # assert cifar_is_unsorted(args, labels)
    # assert feature_is_sorted(labels)


    # ---- first feature sampling
    # calc sample number via sample percent
    sample_num = int(features.shape[0] * args.percent * 0.01)
    # features L2 Norm
    logging.info(f"[main] features has `Neg`: {(features < 0).any()}")  # True
    # logging.info(f"[main] features[0]: {features[0]}")
    # features = features / np.linalg.norm(features, 2, axis=1, keepdims=True)
    logging.info(f'[main] features.dtype: {features.dtype}')  ### [main] features.dtype: float64
    features = features.astype(np.float32)
    logging.info(f'[main] features.dtype: {features.dtype}')  ### [main] features.dtype: float32
    faiss.normalize_L2(features)  # same as above L2 Norm
    logging.info(f"[main] features has `Neg`: {(features < 0).any()}")  # True
    # sys.exit(-1)

    if True:
        # cluster features via KMeans
        indexs, dists = cluster_features(features, sample_num)
        # sample ids nearest to centroids in features
        sample_ids = sample(indexs, dists, sample_num)
    else:
        # cluster_and_sample: instead of `cluster_features` and `sample`, res equal is tested
        sample_ids = cluster_and_sample(features, sample_num)
    

    # ---- val data loading
    val_features, val_label = load_features(dataset, dataset_type='val')
    logging.info(f'[main] val_features.dtype: {val_features.dtype}')
    val_features = val_features.astype(np.float32)
    logging.info(f'[main] val_features.dtype: {val_features.dtype}')
    faiss.normalize_L2(val_features)


    # ---- Base class statistics
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
    
    assert len(base_means) == len(base_data)
    # sys.exit(0)

    # support data and query data
    support_ids = sample_ids
    support_data = features[support_ids]
    support_label = labels[support_ids]
    query_ids = [e for e in range(num_features) if e not in support_ids]
    query_data = features[query_ids]
    query_label = labels[query_ids]
    logging.info(f"[main] support_data has `NaN`: {np.isnan(support_data).any()}")  # False
    logging.info(f"[main] support_data has `Neg`: {(support_data < 0).any()}")  # True

    # ---- Tukey's transform
    beta = 0.5  # support_data and query_data have negative values
    # support_data = np.power(support_data[:, ], beta)  # here support_data[:, ] work? this seems same as support_data or support_data[:]
    # query_data = np.power(query_data[:, ], beta)      # if wanna copy, we can use query_data.copy()
    # NumPy, RuntimeWarning: invalid value encountered in power
    support_data = np.sign(support_data) * np.power(np.abs(support_data), beta)
    query_data = np.sign(query_data) * np.power(np.abs(query_data), beta)
    val_data = np.sign(val_features) * np.power(np.abs(val_features), beta)
    logging.info(f'[main] support_data.shape: {support_data.shape}')
    logging.info(f'[main] query_data.shape: {query_data.shape}')
    logging.info(f'[main] val_data.shape: {val_data.shape}')
    logging.info(f"[main] support_data has `NaN`: {np.isnan(support_data).any()}")  # True -> False
    # sys.exit(-1)
    

    # ---- distribution calibration and second feature sampling
    if not args.no_aug:
        dim = features.shape[1]  # 384
        logging.info(f'[main] dim == {dim}')
        # index_type = 'Flat'
        # metric_type = faiss.METRIC_INNER_PRODUCT
        # index_factory = faiss.index_factory(dim, index_type, metric_type)
        # index_factory = faiss.IndexFlatL2(dim)
        index_factory = faiss.IndexFlatIP(dim)
        if not index_factory.is_trained:
            index_factory.train(features)
        logging.info(f'[main] index.is_trained = {index_factory.is_trained}')  ### [main] index.is_trained = True
        index_factory.add(features)
        logging.info(f'[main] index.ntotal = {index_factory.ntotal}\n')  ### [main] index.ntotal = 130000
        topK = 1  # search topK similarity

        sampled_data = []
        sampled_ids = []
        sampled_label = []
        # num_sampled = 5
        num_sampled = 2
        total_sampled = num_sampled * sample_num
        corrected_num = total_sampled
        fixed_num = 0

        logging.info(f'[main] [Aug] [DC] === start Distribution Calibration and sample Aug === ')
        for i in range(sample_num):
            if args.log_aug:
                logging.info(f'[main] [Aug] [DC] ================ {i} ================ ')
            mean, cov = distribution_calibration(support_data[i], base_means, base_cov, k=1, alpha=0)
            tmp = np.array(base_data[int(support_label[i])])
            mean = np.mean(tmp, axis=0)
            cov = np.cov(tmp.T)
            # logging.info(f'[main] mean: {mean}')
            # logging.info(f'[main] tmp_mean: {tmp_mean}')
            # logging.info(f'[main] mean == tmp_mean: {all(mean==tmp_mean)}')
            sample_virtual = np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled)
            # logging.info(f'[main] mean.shape: {mean.shape}')  ### [main] mean.shape: (384,)
            # logging.info(f'[main] cov.shape: {cov.shape}')    ### [main] cov.shape: (384, 384)
            # logging.info(f'[main] sample_virtual.shape: {sample_virtual.shape}')  ### [main] sample_virtual.shape: (`num_sampled`, 384)
            # logging.info(f'[main] sample_virtual.dtype: {sample_virtual.dtype}')  ### [main] sample_virtual.dtype: float64
            
            sample_virtual = sample_virtual.astype(np.float32)  # must for faiss.normalize_L2
            assert sample_virtual.dtype == np.float32
            faiss.normalize_L2(sample_virtual)
            D, I = index_factory.search(sample_virtual, topK)
            I = I.flatten()
            sample_reality = features[I]

            pseudo_label = [support_label[i]] * num_sampled * topK
            pseudo_index = 0
            for idx in I:
                if labels[idx] != support_label[i]:
                    if args.log_aug:
                        # logging.info(f"[main] [Aug] ---------- {idx}th [{labels[idx]}] Error label [{support_label[i]}] ---------- ")
                        logging.info(f"[main] [Aug] ---------- {idx}th [{labels[idx]}] Error label [{pseudo_label[pseudo_index]}] ---------- ")
                    corrected_num -= 1
                    # uncertainty
                    mean_real = np.mean(np.array(base_data[int(labels[idx])]), axis=0)
                    mean_pred = np.mean(np.array(base_data[int(support_label[i])]), axis=0)
                    distance1 = np.linalg.norm(features[idx] - mean_real)
                    distance2 = np.linalg.norm(features[idx] - mean_pred)
                    if args.uncertainty and distance1 <= distance2:
                        # logging.info(f"[main] [Aug] uncertainty is added. ")
                        if args.log_aug:
                            logging.info(f"[main] [Aug] ++++++++++ {idx}th Error label will be fixed ++++++++++ ")
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

        logging.info(f"[main] sample corrected: {corrected_num} / {total_sampled} ({corrected_num / total_sampled * 100} %)")
        logging.info(f"[main] sample fixed num: {fixed_num}")
        # logging.info(f"[main] support_data has `NaN`: {np.isnan(support_data).any()}")  # True -> False
        # logging.info(f"[main] sampled_data has `NaN`: {np.isnan(sampled_data).any()}")  # False
        # sys.exit(-1)

    # ---- train classifier
    from sklearn.linear_model import LogisticRegression as LR
    logging.info(f"[main] -------------------------------------")
    logging.info(f"[main] classifier building...")
    
    if args.no_aug:
        logging.info(f"[main] classifier w.o. DC Aug")
        classifier = LR(max_iter=1000).fit(X=support_data, y=support_label)  # base LR classifier have no DC Aug
    else:
        logging.info(f"[main] classifier with DC Aug")
        classifier = LR(max_iter=1000).fit(X=X_aug, y=Y_aug)

    if args.val:
        logging.info(f"[main] predict `val` dataset")
        query_data, query_label = val_data, val_label
    else:
        logging.info(f"[main] predict `query` dataset")

    logging.info(f"[main] Start LR classifer predict...")
    predicts = classifier.predict(query_data)
    logging.info(f"[main] LR classifer predict done.")

    acc = np.mean(predicts == query_label)
    acc_list = []
    acc_list.append(acc)
    logging.info(f"[main] @Acc1 (query): {float(np.mean(acc_list))}")
    logging.info(f"[main] @Acc1 (query): {float(np.mean(acc_list)) * 100} %")




"""
python sample_tools/distribution_calibration.py -p 1 --uncertainty
---------------------
[2023-09-01 16:51:51,556] [INFO] ### [main] [Aug] [DC] === start Distribution Calibration and sample Aug === 
[2023-09-01 16:55:26,764] [INFO] ### [main] sample corrected: 2362 / 2600 (90.84615384615384 %)
[2023-09-01 16:55:26,764] [INFO] ### [main] sample fixed num: 98
[2023-09-01 16:55:43,657] [INFO] ### [main] predict `query` dataset
[2023-09-01 16:55:43,657] [INFO] ### [main] Start LR classifer predict...
[2023-09-01 16:55:44,120] [INFO] ### [main] LR classifer predict done.
[2023-09-01 16:55:44,121] [INFO] ### [main] @Acc1 (query): 0.8458000202015493
[2023-09-01 16:55:44,121] [INFO] ### [main] @Acc1 (query): 84.58000202015494 %
---------------------
1300 + __ --> 84.64%
1300 + __ --> 84.52%
1300 + 98 --> 84.58%
========================================================================
python sample_tools/distribution_calibration.py -p 1 --val --uncertainty
---------------------
[2023-09-01 16:33:59,090] [INFO] ### [main] [Aug] [DC] === start Distribution Calibration and sample Aug === 
[2023-09-01 16:37:34,562] [INFO] ### [main] sample corrected: 2377 / 2600 (91.42307692307692 %)
[2023-09-01 16:37:34,562] [INFO] ### [main] sample fixed num: 83
[2023-09-01 16:55:43,657] [INFO] ### [main] predict `val` dataset
[2023-09-01 16:37:54,401] [INFO] ### [main] Start LR classifer predict...
[2023-09-01 16:37:54,463] [INFO] ### [main] LR classifer predict done.
[2023-09-01 16:37:54,464] [INFO] ### [main] @Acc1 (query): 0.8144
[2023-09-01 16:37:54,464] [INFO] ### [main] @Acc1 (query): 81.44 %
---------------------
1300 + 83 --> 81.44%
1300 + __ --> 81.46%
1300 + 90 --> 81.06%
1300 +102 --> 81.80%
========================================================================
python sample_tools/distribution_calibration.py -p 1 --val --no_aug
---------------------
[2023-09-01 17:30:35,356] [INFO] ### [main] classifier w.o. DC Aug
[2023-09-01 17:30:38,717] [INFO] ### [main] predict `val` dataset
[2023-09-01 17:30:38,718] [INFO] ### [main] Start LR classifer predict...
[2023-09-01 17:30:38,777] [INFO] ### [main] LR classifer predict done.
[2023-09-01 17:30:38,781] [INFO] ### [main] @Acc1 (query): 0.797
[2023-09-01 17:30:38,781] [INFO] ### [main] @Acc1 (query): 79.7 %
---------------------
79.70 * 2, beta = 0.5; ( 79.78 when sample v0 is True )
77.04 * 2, beta = 1.0;
========================================================================
python sample_tools/distribution_calibration.py -p 1 --val
---------------------
[2023-09-01 18:20:47,172] [INFO] ### [main] [Aug] [DC] === start Distribution Calibration and sample Aug === 
[2023-09-01 18:24:23,204] [INFO] ### [main] sample corrected: 2383 / 2600 (91.65384615384615 %)
[2023-09-01 18:24:23,204] [INFO] ### [main] sample fixed num: 0
[2023-09-01 18:24:32,115] [INFO] ### [main] -------------------------------------
[2023-09-01 18:24:32,115] [INFO] ### [main] classifier building...
[2023-09-01 18:24:32,115] [INFO] ### [main] classifier with DC Aug
[2023-09-01 18:24:35,624] [INFO] ### [main] predict `val` dataset
[2023-09-01 18:24:35,625] [INFO] ### [main] Start LR classifer predict...
[2023-09-01 18:24:35,653] [INFO] ### [main] LR classifer predict done.
[2023-09-01 18:24:35,656] [INFO] ### [main] @Acc1 (query): 0.8074
[2023-09-01 18:24:35,657] [INFO] ### [main] @Acc1 (query): 80.74 %
---------------------
1300 + 0 --> 80.74
"""
