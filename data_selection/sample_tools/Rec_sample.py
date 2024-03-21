import os
import math
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from temp_model_arch import MyNet, Sample_Attn_RecNet, BSNET_Conv

import logging
logging.basicConfig(format='[%(asctime)s] [%(levelname)s] ### %(message)s', level=logging.INFO)


def load_features(args, is_train=True):
    logging.info("[load_features] loading Start...")
    file_path = args.trn_data_path if is_train else args.val_data_path
    input = np.load(file_path)
    # features, _ = input[:, :-1], input[:, -1]
    features, labels = input[:, :-1], input[:, -1]
    features = features.astype(np.float32)
    assert features.dtype == np.float32
    logging.info("[load_features] loading Done...")
    logging.info(f'[load_features] features.shape: {features.shape}')      # features.shape: (50000, 384)
    logging.info(f'[load_features] labels.shape: {labels.shape}')          # labels.shape: (50000,)
    return features, labels


def dataset_info(labels, name='train_dataset'):
    num_features = len(labels)
    num_classes  = len(set(labels))
    num_per_class= num_features // num_classes
    logging.info(f"[dataset_info] [{name}] number of features: {num_features}")
    logging.info(f"[dataset_info] [{name}] number of classes : {num_classes}")
    logging.info(f"[dataset_info] [{name}] number per class  : {num_per_class}")
    return num_features, num_classes, num_per_class


def json_dump(args, sample_ids):
    logging.info(f'[json_dump] start...')
    sample_num = len(sample_ids)
    logging.info(f'[json_dump] len of sample: {sample_num}')
    if args.filename is None:
        name = args.trn_data_path.split("/")[-1]
        name = name[:-4]
        args.filename = name + "_Rec_sampleNum_%d.json" % (sample_num)
    output_path = os.path.join(args.output_dir, args.filename)
    logging.info(f'[json_dump] output dir: {output_path}')

    with open(output_path, "w") as f:
        json.dump(sample_ids, f)
    logging.info(f'[json_dump] done.')


def resume_sample_ids(args):
    if os.path.isfile(args.sample_resume) and args.sample_resume.endswith('npy'):
        logging.info(f"[resume_sample_ids] resume sample ids from npy file")
        sample_ids = np.load(args.sample_resume)
    elif os.path.isfile(args.sample_resume) and args.sample_resume.endswith('json'):
        logging.info(f"[resume_sample_ids] resume sample ids from json file")
        with open(args.sample_resume, 'r') as f:
            sample_ids = json.load(f)
    else:
        logging.info(f"[resume_sample_ids] Resume Failed.")
        return None
    return sample_ids


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


def acc_via_classifier(args, trn_X, trn_y, val_X, val_y):
    
    logging.info(f" ---------------------------------------------------")
    logging.info(f"[LogisticRegression] classifier is building...")
    classifier = LR(max_iter=1000).fit(X=trn_X, y=trn_y)
    logging.info(f"[LogisticRegression] classifier is builded.")

    logging.info(f"[LogisticRegression] classifier predict start...")
    predicts = classifier.predict(val_X)
    logging.info(f"[LogisticRegression] classifier predict done.")

    acc_score = accuracy_score(val_y, predicts)
    logging.info(f"[LogisticRegression] @Acc1 : {acc_score}")
    logging.info(f"[LogisticRegression] @Acc1 : {acc_score * 100} %")
    return acc_score


def cls_LR(features, labels, sample_ids, args):
    num_features = features.shape[0]
    sample_num = int(num_features * args.percent * 0.01)
    assert len(sample_ids) == sample_num, "len of sample_ids not equals to sample_num"

    # trn data
    trn_data = features[sample_ids]
    trn_label = labels[sample_ids]
    # val data
    val_data, val_label = load_features(args, is_train=False)
    val_data = val_data / np.linalg.norm(val_data, 2, axis=1, keepdims=True)
    # ---- Tukey's transform
    trn_data = Tukey_Transform(trn_data, beta=args.beta, info="trn_data")
    val_data = Tukey_Transform(val_data, beta=args.beta, info="val_data")

    # ---- train classifier
    test_acc = acc_via_classifier(args, trn_data, trn_label, val_data, val_label)
    logging.info(f"[cls_LR] @Acc1 (val): {test_acc * 100} %")



def train_one_epoch(model, optimizer, trn_data, epoch, args):
    model.train()
    
    weight, output =  model(trn_data)
    loss = F.smooth_l1_loss(output, trn_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    L1 = loss.item()
    # l1_list.append(L1)
    logging.info('[train_one_epoch] Epoch: {}\tLoss: {:.6f}'.format(epoch, L1))

    if epoch == args.epoch_num:
        weight = weight.detach().cpu().numpy().flatten()
        # channel_weight_list.append(weight)
        top = int(trn_data.shape[1] * args.percent * 0.01)
        sample_ids = weight.argsort()[-top:][::-1]
        # logging.info('[train_one_epoch] ids sampled by weight ->'.format(top),list(sample_ids))
        return sorted(sample_ids.tolist())
    return None


def train(trn_data, args):

    total_num = trn_data.shape[0]
    sample_num = int(total_num * args.percent * 0.01)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'[Data] [start] trn_data ---> {device}')
    trn_data = torch.from_numpy(trn_data)[None].to(device)  # or unsqueeze, reshape, view
    logging.info(f'[Data] [done ] trn_data ---> {device}')

    logging.info(f'[Model] [start] create model')
    # model = MyNet(total_num, sample_num).to(device)
    # model = BSNET_Conv(total_num, sample_num).to(device)
    model = Sample_Attn_RecNet(total_num, sample_num)  # Pipeline Model Parallel
    logging.info(f'[Model] [done ] create model')

    mem_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'[Model] number of params: {mem_params}')
    # mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem_bufs = sum([buf.numel() for buf in model.buffers()])
    logging.info(f'[Model] number of buffers: {mem_bufs}')  # 0
    mem = mem_params + mem_bufs
    # mem = math.ceil(mem / 1024 / 1024)
    mem = -(-mem // (1024 * 1024))
    logging.info(f'[Model] Memory needed: {mem}M')  # 6396M -> 1880M -> 157M
    # return None

    # model = torch.nn.DataParallel(model)  # Pipeline Model Parallel instead of Data Parallel.

    optimizer = torch.optim.AdamW(model.parameters())
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.CosineSimilarity()
    # criterion = nn.PairwiseDistance(p=2)

    logging.info(f'[Train] [start] training...')
    sample_ids = None
    for epoch in range(1, args.epoch_num + 1):
        sample_ids = train_one_epoch(model, optimizer, trn_data, epoch, args)
        # val(model, val_data)
    logging.info(f'[Train] [done ] trained.')
    return sample_ids

"""
@torch.no_grad()
def val(model, val_data):
    # model.eval()
    weight, output =  model(val_data)
    loss = F.smooth_l1_loss(output, val_data).item()
    logging.info('[val] Eval loss: {:.6f}'.format(loss))
"""


def reconstruct_sample(trn_data, args):
    # weight rec sample
    logging.info(f'[Rec sample] start...')
    sample_ids = train(trn_data, args)
    logging.info(f'[Rec sample] done.')

    # std output
    if sample_ids is not None:
        logging.info(sample_ids)
        json_dump(args, sample_ids)
    else:
        logging.info(f"[Sample Failed] sample_ids is None")



def main(args):

    # load_features && features L2 Norm
    trn_data, labels = load_features(args)
    trn_data = trn_data / np.linalg.norm(trn_data, 2, axis=1, keepdims=True)
    dataset_info(labels)
    sample_ids = resume_sample_ids(args)

    if sample_ids:
        cls_LR(trn_data, labels, sample_ids, args)
        logging.info(f'[main] just CLS, do NOT sample.')
        return 0

    reconstruct_sample(trn_data, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    parser.add_argument('-trn', '--trn_data_path', default='features/CIFAR10_train.npy', type=str,help='path of saved features')
    parser.add_argument('-val', '--val_data_path', default='features/CIFAR10_val.npy', type=str,help='path of saved features')
    parser.add_argument('--output_dir', default='features', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default=None, type=str, help='filename of the visualization')
    parser.add_argument('-p', '--percent', default=0.5, type=float, help='sample percent')
    parser.add_argument('--beta', default=0.5, type=float, help='Tukey_Transform hyperparam beta')
    parser.add_argument('-e', '--epoch_num', default=100, type=int, help='sample percent')
    parser.add_argument('-r', '--sample_resume', default='', type=str, help='first sample ids file path')
    args = parser.parse_args()
    main(args)



"""
default CIFAR10
++++++++++++++++++
train and sample
-----------------
...
[2023-09-28 17:00:44,238] [INFO] ### [train_one_epoch] Epoch: 1 Loss: 0.126220
[2023-09-28 17:00:44,347] [INFO] ### [train_one_epoch] Epoch: 2 Loss: 0.125672
[2023-09-28 17:00:44,448] [INFO] ### [train_one_epoch] Epoch: 3 Loss: 0.119842
[2023-09-28 17:00:44,547] [INFO] ### [train_one_epoch] Epoch: 4 Loss: 0.079540
[2023-09-28 17:00:44,643] [INFO] ### [train_one_epoch] Epoch: 5 Loss: 0.024490
[2023-09-28 17:00:44,738] [INFO] ### [train_one_epoch] Epoch: 6 Loss: 0.007928
[2023-09-28 17:00:44,834] [INFO] ### [train_one_epoch] Epoch: 7 Loss: 0.002792
[2023-09-28 17:00:44,932] [INFO] ### [train_one_epoch] Epoch: 8 Loss: 0.001490
[2023-09-28 17:00:45,027] [INFO] ### [train_one_epoch] Epoch: 9 Loss: 0.001340
[2023-09-28 17:00:45,123] [INFO] ### [train_one_epoch] Epoch: 10        Loss: 0.001312
[2023-09-28 17:00:45,219] [INFO] ### [train_one_epoch] Epoch: 11        Loss: 0.001302
...
[2023-09-28 17:00:49,278] [INFO] ### [train_one_epoch] Epoch: 50        Loss: 0.001302
[2023-09-28 17:00:49,280] [INFO] ### [Train] [done ] trained.
[2023-09-28 17:00:49,280] [INFO] ### [json_dump] len of sample: 250
[2023-09-28 17:00:49,281] [INFO] ### [json_dump] output dir: features/CIFAR10_train_Rec_sampleNum_250.json

====================================================================================================================

resume and classification
-----------------
...
[2023-09-28 18:00:35,199] [INFO] ### [cls_LR] @Acc1 (val): 89.19 %
[2023-09-28 18:00:35,199] [INFO] ### [main] just CLS, do NOT sample.
"""


"""
CIFAR100
++++++++++++++++++
train and sample
-----------------
...
[2023-09-28 18:17:05,703] [INFO] ### [train_one_epoch] Epoch: 1 Loss: 0.126214
[2023-09-28 18:17:05,890] [INFO] ### [train_one_epoch] Epoch: 2 Loss: 0.124627
[2023-09-28 18:17:06,065] [INFO] ### [train_one_epoch] Epoch: 3 Loss: 0.010057
[2023-09-28 18:17:06,241] [INFO] ### [train_one_epoch] Epoch: 4 Loss: 0.001344
[2023-09-28 18:17:06,415] [INFO] ### [train_one_epoch] Epoch: 5 Loss: 0.001302
...
[2023-09-28 18:17:14,597] [INFO] ### [train_one_epoch] Epoch: 50        Loss: 0.001302
[2023-09-28 18:17:14,617] [INFO] ### [Train] [done ] trained.
[2023-09-28 18:17:14,618] [INFO] ### [json_dump] len of sample: 1000
[2023-09-28 18:17:14,618] [INFO] ### [json_dump] output dir: features/CIFAR100_train_Rec_sampleNum_1000.json

====================================================================================================================

resume and classification
-----------------
...
[2023-09-28 18:22:57,290] [INFO] ### [cls_LR] @Acc1 (val): 61.370000000000005 %
[2023-09-28 18:22:57,291] [INFO] ### [main] just CLS, do NOT sample.
"""


"""
ImageNet
++++++++++++++++++
train and sample, epoch 100
-----------------
...
[2023-09-28 18:41:58,196] [INFO] ### [train_one_epoch] Epoch: 1 Loss: 0.126307
[2023-09-28 18:41:58,501] [INFO] ### [train_one_epoch] Epoch: 2 Loss: 0.125839
[2023-09-28 18:41:58,829] [INFO] ### [train_one_epoch] Epoch: 3 Loss: 0.125163
...
[2023-09-28 18:42:29,067] [INFO] ### [train_one_epoch] Epoch: 98        Loss: 0.001402
[2023-09-28 18:42:29,368] [INFO] ### [train_one_epoch] Epoch: 99        Loss: 0.001402
[2023-09-28 18:42:29,736] [INFO] ### [train_one_epoch] Epoch: 100       Loss: 0.001402
[2023-09-28 18:42:29,740] [INFO] ### [Train] [done ] trained.
[2023-09-28 18:42:29,741] [INFO] ### [json_dump] len of sample: 1300
[2023-09-28 18:42:29,741] [INFO] ### [json_dump] output dir: features/ImageNet_train_Rec_sampleNum_1300.json

====================================================================================================================

resume and classification
-----------------
...
[2023-09-28 18:45:53,337] [INFO] ### [cls_LR] @Acc1 (val): 78.8 %
[2023-09-28 18:45:53,338] [INFO] ### [main] just CLS, do NOT sample.
"""
