from train import train
from data import load_data_chemprot_re, load_mole_data
import torch
import os
# from train_utils import setup_common
# from utils import mkdir
# from model.load_model import get
# from transformers import BertTokenizer
from transformers import AutoTokenizer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
from options import read_args
from train_utils import *

from utils import dump_file, load_file
if __name__ == '__main__':
    args = read_args()
    if args.debug:
        # args.plm="bert-ba"
        # args.use_amp=False
        # args.use_cache=False
        # args.use_cache=True
        # data
        data_dir = "data_online/ChemProt_Corpus/chemprot_preprocessed/"
        args.train_file = data_dir + "train.txt"
        args.val_file = data_dir + "dev.txt"
        args.test_file = data_dir + "test.txt"
        args.model_name = "re_model"
        args.exp = "re"
        # args.num_epoch = 300
        # args.batch_size = 4
        args.burn_in = 1
        # args.lr = 1e-4
        # args.grad_accumulation_steps = 32
        # args.plm="prajjwal1/bert-medium"
        # args.plm="prajjwal1/bert-tiny"
        args.plm = "allenai/scibert_scivocab_uncased"
        # args.patience = 8
        # args.g_dim = 128
        # args.bert_only=1
        # args.model_type="tdg"
        # args.t_only=True
        # args.g_dim=32
        # args.use_cache=False

    set_seeds(args)
    print("tokenizer1")
    if not os.path.exists("tokenizer.pkl"):
        tokenizer = AutoTokenizer.from_pretrained(args.plm)
        dump_file(tokenizer, "tokenizer.pkl")
    tokenizer=load_file("tokenizer.pkl")
    #

    print("tokenizer2")
    train_data, val_data, test_data = load_data_chemprot_re(args, args.train_file, tokenizer), \
                                      load_data_chemprot_re(args, args.val_file, tokenizer), \
                                      load_data_chemprot_re(args, args.test_file, tokenizer)

    # load model and data etc.
    args, model, optimizer = setup_common(args)

    # train_data, val_data, test_data = load_data(args.train_file), load_data(args.val_file), load_data(args.test_file)
    train(args, model, optimizer, (train_data, val_data, test_data))
