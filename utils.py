# import torch
import sys
import re
import time
# import torch
from pprint import pprint as pp
import pickle as pkl
from collections import deque, defaultdict, Counter
import json
from IPython import embed
import random
import time
import os
import numpy as np
import argparse
import gc
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pynvml import *
from model.load_model import get_model, load_model_from_path
from data import *
from options import *
from torch import optim
# import wandb
import logging


def is_symmetric(g):
    return np.sum(np.abs(g.T - g)) == 0


def get_logger(args):
    if os.path.exists(args.experiment_path + args.experiment + ".txt"):

        with open(args.experiment_path + "count.txt", mode="r") as f:
            pos = int(f.readlines()[-1].strip())
        with open(args.experiment_path + "count.txt", mode="w") as f:
            f.write(str(pos + 1))
        os.rename(args.experiment_path + args.experiment + ".txt",
                  args.experiment_path + args.experiment + "_" + str(pos) + ".txt")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    mkdir(args.experiment_path)
    output_file_handler = logging.FileHandler(args.experiment_path + args.experiment + ".txt")
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    return logger


def setup_common():
    args = read_args()
    # wandb.config.update(args)
    set_seeds(args)
    args.device = gpu_setup(use_gpu=args.use_gpu)
    # wandb.init(config=args, project=args.experiment)
    if args.debug:
        # args.plm="bert-ba"
        args.use_amp=False
        args.num_epoch = 300
        args.batch_size = 64
        args.burn_in = 1
        args.lr=1e-3
        args.grad_accumulation_steps = 1
        args.plm="prajjwal1/bert-tiny"
    model = get_model(args)
    # print("model", model)
    # view_model_param(args, model)

    downstream_layers = ["extractor", "bilinear", "combiner", "gnn", "msg_encoder", "query_encoder"]
    optimizer = get_optimizer(args, model, downstream_layers)

    model, optimizer, args.start_epoch, args.best_dev_score = load_model_from_path(model, optimizer,
                                                                                   args.model_path,
                                                                                   gpu=args.use_gpu)
    return args, model, optimizer


def gpu_setup(use_gpu=True, gpu_id=0, use_random_available=True):
    print("Setting up GPU")
    if not use_random_available:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print(f"{torch.cuda.device_count()} GPU available")
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")

    else:
        if not torch.cuda.is_available():
            print('cuda not available')
        device = torch.device("cpu")
    return device


def view_model_param(args, model):
    # model = get_model(args)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', args.model_name, total_param)
    return total_param


def get_optimizer(args, model, downstream_layers):
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in downstream_layers)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in downstream_layers)],
         "lr": args.lr}
    ]
    return optim.AdamW(optimizer_grouped_parameters,
                       lr=args.lr,
                       weight_decay=args.weight_decay,
                       eps=args.adam_epsilon)


def to_tensor_float(data):
    return torch.as_tensor(data, dtype=torch.float)


def to_tensor_long(data):
    return torch.as_tensor(data, dtype=torch.long)


def get_tensor_info(tensor):
    return f"Shape: {tensor.shape} | Type: {tensor.type()} | Device: {tensor.device}"


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def join(str1, str2):
    return os.path.join(str1, str2)


def get_ext(filename):
    return os.path.splitext(filename)[1]


def dump_file(obj, filename):
    if get_ext(filename) == ".json":
        with open(filename, "w+") as w:
            json.dump(obj, w)
    elif get_ext(filename) == ".pkl":
        with open(filename, "wb+") as w:
            pkl.dump(obj, w)
    else:
        print("not pkl or json")
        with open(filename, "w+", encoding="utf-8") as w:
            w.write(obj)


def load_file(filename):
    if get_ext(filename) == ".json":
        with open(filename, "r") as r:
            res = json.load(r)
    elif get_ext(filename) == ".pkl":
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res


def mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
