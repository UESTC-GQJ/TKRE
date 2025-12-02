import torch
import random
import os
import pdb
import time
import logging
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    output = (input_ids, input_mask, labels, ss, os)
    return output


def loadLogger(work_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)

    # work_dir = os.path.join(origin_work_dir,
    #                         time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)

    logger.addHandler(fHandler)

    return logger


def find_best_performance(result_output):
    assert len(result_output) >= 1
    all_metric_list = list(result_output[0].keys())
    metric_list = []
    for metric in all_metric_list:
        if 'dev' in metric:
            metric_list.append(metric)
    best_metric_res = {}
    best_metric_epoch = {}
    for metric in metric_list:
        best_metric_res[metric] = result_output[0][metric]
        best_metric_epoch[metric] = 0
        
    for epoch in result_output:
        for metric in metric_list:
            # pdb.set_trace()
            if result_output[epoch][metric] > best_metric_res[metric]:
                best_metric_res[metric] = result_output[epoch][metric]
                best_metric_epoch[metric] = epoch
    
    corresponding_res = {}
    for metric in metric_list:
        corresponding_res[metric] = result_output[best_metric_epoch[metric]]
    # pdb.set_trace()
    return best_metric_epoch, corresponding_res