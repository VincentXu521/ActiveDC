import json
import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F

import logging
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] ### %(message)s', level=logging.INFO)

# conda activate active_finetune (3090, faiss-gpu==1.7.4)
# export LD_LIBRARY_PATH=/gpfsdata/home/wenshuai/miniconda3/envs/active_finetune/lib:$LD_LIBRARY_PATH
import faiss



def load_features(args):
    logging.info("[load_features] loading Start...")
    input = np.load(args.feature_path)
    # features, _ = input[:, :-1], input[:, -1]
    features, labels = input[:, :-1], input[:, -1]
    logging.info("[load_features] loading Done...")
    logging.info(f'[load_features] features.shape: {features.shape}')      # features.shape: (130000, 384)
    logging.info(f'[load_features] labels.shape: {labels.shape}')          # labels.shape: (130000,)
    return features, labels



def json_dump(args, sample_ids):
    logging.info(f'[json_dump] start...')
    sample_num = len(sample_ids)
    logging.info(f'[json_dump] len of sample: {sample_num}')
    if args.filename is None:
        name = args.feature_path.split("/")[-1]
        name = name[:-4]
        args.filename = name + "_PCA_sorted_sampleNum_%d.json" % (sample_num)
    output_path = os.path.join(args.output_dir, args.filename)
    logging.info(f'[json_dump] output dir: {output_path}')

    with open(output_path, "w") as f:
        json.dump(sample_ids, f)
    logging.info(f'[json_dump] done.')



def main(args):

    # # load_features
    features, labels = load_features(args)  # features.shape: (130000, 384)

    DIM = 384
    total_num = features.shape[0]
    sample_num = int(total_num * args.percent * 0.01)
    cls_num = 100
    num_per = total_num // cls_num

    # features L2 Norm
    features = features.astype(np.float32)
    faiss.normalize_L2(features)
    # features = features / np.linalg.norm(features, 2, axis=1, keepdims=True)

    # convert features shape
    # features_T = features.T

    # PCA 
    start = 0
    tr_T_concat = np.random.random((1, DIM))
    while start + num_per <= total_num:
        mat = faiss.PCAMatrix(num_per, sample_num // cls_num)
        features_T = features[start: start + num_per].T
        mat.train(features_T)
        assert mat.is_trained
        tr = mat.apply_py(features_T)
        logging.info(f"[PCA] features_T.shape: {features_T.shape}")  # features_T.shape: (384, 1300)
        logging.info(f"[PCA] tr.shape: {tr.shape}")                  # tr.shape:         (384, 13)
        tr_T_concat = np.concatenate([tr_T_concat, tr.T.copy()])
        start += num_per

    tr_T_concat = tr_T_concat[1:]
    assert tr_T_concat.shape == (sample_num, DIM)

    # index
    # index_flat = faiss.IndexFlatL2(DIM)
    index_flat = faiss.IndexFlatIP(DIM)
    if args.gpu:
        index_flat = faiss.index_cpu_to_all_gpus(index_flat)
        logging.info(f'[Index] [GPU] faiss GPU index is used.')
    if not index_flat.is_trained:
        index_flat.train(features)
    logging.info(f'[Index] index.is_trained = {index_flat.is_trained}')
    index_flat.add(features)
    # faiss.normalize_L2(tr_T_concat)
    D, I = index_flat.search(tr_T_concat, 1)
    I = I.flatten()
    
    sample_ids = sorted(I.tolist())
    logging.info(f"[Index] len of sample_ids: {len(sample_ids)}")
    assert len(sample_ids) == sample_num

    # std output
    logging.info(sample_ids)
    json_dump(args, sample_ids)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    parser.add_argument('--feature_path', default='features/ImageNet_train.npy', type=str,help='path of saved features')
    parser.add_argument('--output_dir', default='features', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default=None, type=str, help='filename of the visualization')
    parser.add_argument('--percent', default=1, type=float, help='sample percent')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    args = parser.parse_args()
    main(args)



"""
    # PCA demo
    # random training data 
    mt = np.random.rand(1000, 40).astype('float32')
    logging.info(f"mt.shape: {mt.shape}")
    mat = faiss.PCAMatrix(40, 10)
    mat.train(mt)
    assert mat.is_trained
    tr = mat.apply_py(mt)
    logging.info(f"tr.shape: {tr.shape}")
    logging.info(f"mt.shape: {mt.shape}")
    # print this to show that the magnitude of tr's columns is decreasing
    print((tr ** 2).sum(0))
    # ----------------------------------
    # output:
    # [2023-09-15 16:33:31,541] [INFO] ### Loading faiss with AVX2 support.
    # [2023-09-15 16:33:31,728] [INFO] ### Successfully loaded faiss with AVX2 support.
    # [2023-09-15 16:33:31,751] [INFO] ### mt.shape: (1000, 40)
    # [2023-09-15 16:33:31,753] [INFO] ### tr.shape: (1000, 10)
    # [2023-09-15 16:33:31,753] [INFO] ### mt.shape: (1000, 40)
    # [115.6147   111.88475  111.06351  108.81746  106.26271  104.6537
    #  101.33385   99.684525  98.89854   96.57537 ]
"""
