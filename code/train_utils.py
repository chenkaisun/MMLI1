import logging
import os
import random
import sys

import numpy as np
import torch
from torch import optim
from transformers import AutoTokenizer

from model.load_model import get_model, load_model_from_path
# from options import read_args
from utils import mkdir, dump_file, load_file
from torch.utils.tensorboard import SummaryWriter
import numpy
from pynvml import *
import torch.nn as nn
import pickle as pkl
from IPython import embed


class ScoreRecorder:
    def __init__(self, path):
        pass

    def get_highest(self):
        pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_logger(args):
    # if os.path.exists(args.experiment_path + args.experiment + ".txt"):
    #
    #     with open(args.experiment_path + "count.txt", mode="r") as f:
    #         pos = int(f.readlines()[-1].strip())
    #     with open(args.experiment_path + "count.txt", mode="w") as f:
    #         f.write(str(pos + 1))
    #     os.rename(args.experiment_path + args.experiment + ".txt",
    #               args.experiment_path + args.experiment + "_" + str(pos) + ".txt")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    mkdir(args.experiment_path)
    output_file_handler = logging.FileHandler(args.experiment_path + args.exp + "_" + args.exp_id + ".txt")
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    return logger


def get_plm_fullname(abbr):
    plm_dict = {
        "base": "bert-base-cased",
        "sap": "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        "sci": "allenai/scibert_scivocab_uncased",
        "tiny": "prajjwal1/bert-tiny"
    }
    return plm_dict[abbr]


def setup_common(args, pretrained_weights=None):
    # args = read_args()
    # wandb.config.update(args)
    # for model running
    mkdir("model")
    mkdir("model/states")

    # args.model_path = "model/states/best_dev_" + args.exp_id + ".pt"

    # set_seeds(args)
    args.device = gpu_setup(use_gpu=args.use_gpu, gpu_id=args.gpu_id, use_random_available=False)
    if "cpu" in str(args.device): args.use_amp = 0

    # # wandb.init(config=args, project=args.experiment)

    if "-tiny" in args.plm:
        args.plm_hidden_dim = 128
    elif "-mini" in args.plm:
        args.plm_hidden_dim = 256
    elif "-small" in args.plm:
        args.plm_hidden_dim = 512
    elif "-medium" in args.plm:
        args.plm_hidden_dim = 512
    else:
        args.plm_hidden_dim = 768

    model = get_model(args, pretrained_weights=pretrained_weights)
    view_model_param(args, model)

    # downstream_layers = ["combiner", "gnn", "cm_attn", 'gnn', 'the_zero','the_one']

    optimizer = get_optimizer(args, model, args.downstream_layers)
    # print(model.named_parameters())
    # print("model", model)
    print("optimizer built")
    model, optimizer, args.start_epoch, args.best_dev_score = load_model_from_path(model, optimizer, args)
    print("model moved to gpu")

    args.logger = get_logger(args)
    args.writer = SummaryWriter(log_dir=args.experiment_path + args.exp + "/")
    args.logger.debug("=====begin of args=====")

    arg_dict = vars(args)
    for key in args.useful_params:
        args.logger.debug(f"{key}: {arg_dict[key]}")
    args.logger.debug("=====end of args=====")

    return args, model, optimizer

def load_word_embed(path: str,
                    dimension: int,
                    *,
                    skip_first: bool = False,
                    freeze: bool = False,
                    sep: str = ' ',
                    file_vocab_path=''):
    """Load pre-trained word embeddings from file.

    Args:
        path (str): Path to the word embedding file.
        skip_first (bool, optional): Skip the first line. Defaults to False.
        freeze (bool, optional): Freeze embedding weights. Defaults to False.

    Returns:
        Tuple[nn.Embedding, Dict[str, int]]: The first element is an Embedding
        object. The second element is a word vocab, where the keys are words and
        values are word indices.
    """
    vocab = {'$$$UNK$$$': 0, '$$$PAD$$$': 1}
    embed_matrix = [[0.0] * dimension]
    file_vocab = pkl.load(open(file_vocab_path, "rb")) if file_vocab_path else None
    # print("file_vocab type", type(file_vocab))
    with open(path, encoding='utf-8') as r:
        if skip_first:
            r.readline()
        for line in r:
            segments = line.rstrip('\n').rstrip(' ').split(sep)
            word = segments[0]
            if (not file_vocab) or word in file_vocab:
                vocab[word] = len(vocab)
                # print("segments[1:]", len(segments[1:]))
                embed = [float(x) for x in segments[1:]]
                embed_matrix.append(embed)
    print('Loaded %d word embeddings' % (len(embed_matrix) - 1))

    embed_matrix = torch.FloatTensor(embed_matrix)

    word_embed = nn.Embedding.from_pretrained(embed_matrix,
                                              freeze=freeze,
                                              padding_idx=0)
    return word_embed, vocab
def gpu_setup(use_gpu=True, gpu_id=2, use_random_available=True):
    print("Setting up GPU")
    if not use_random_available:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_gpus = 1
    if torch.cuda.is_available() and use_gpu:
        print(f"{torch.cuda.device_count()} GPU available")
        # print('cuda available with GPU:', torch.cuda.get_device_name(0))

        # device = torch.device("cuda")
        device = torch.device("cuda:"+str(gpu_id))

    else:
        if not torch.cuda.is_available():
            print('cuda not available')
        device = torch.device("cpu")
    print("device", device)
    return device


def view_model_param(args, model):
    # model = get_model(args)
    total_param = 0
    print(model)
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
                       lr=args.plm_lr,
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tokenizer(plm, save_dir="tokenizer/"):
    return AutoTokenizer.from_pretrained(plm)
    mkdir(save_dir)
    tk_name = plm.split("/")[-1].replace("-", "_") + "_tokenizer.pkl"
    tk_name = os.path.join(save_dir, tk_name)
    if not os.path.exists(tk_name):
        tokenizer = AutoTokenizer.from_pretrained(plm)
        dump_file(tokenizer, tk_name)
    return load_file(tk_name)


def get_gpu_mem_info():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')
