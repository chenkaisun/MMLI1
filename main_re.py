from train import train
from data import load_data_chemprot_re, load_mole_data
import torch
import os
# from train_utils import setup_common
# from utils import mkdir
# from model.load_model import get
# from transformers import BertTokenizer
from transformers import AutoTokenizer
from options import read_args
from train_utils import *

from utils import dump_file, load_file
from pprint import pprint as pp
from sklearn.metrics import f1_score
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

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
        args.burn_in = 1
        args.plm = "allenai/scibert_scivocab_uncased"
        # args.num_epoch = 300
        # args.batch_size = 4
        # args.lr = 1e-4
        # args.grad_accumulation_steps = 32
        # args.plm="prajjwal1/bert-medium"
        # args.plm="prajjwal1/bert-tiny"
        # args.patience = 8
        # args.g_dim = 128
        # args.bert_only=1
        # args.model_type="tdg"
        # args.t_only=True
        # args.g_dim=32
        # args.use_cache=False

    set_seeds(args)
    print("tokenizer1")
    tk_name = args.plm.split("/")[-1].replace("-", "_") + "_tokenizer.pkl"
    if not os.path.exists(tk_name):
        tokenizer = AutoTokenizer.from_pretrained(args.plm)
        dump_file(tokenizer, tk_name)
    tokenizer = load_file(tk_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.plm)

    print("tokenizer2")
    train_data, val_data, test_data = load_data_chemprot_re(args, args.train_file, tokenizer), \
                                      load_data_chemprot_re(args, args.val_file, tokenizer), \
                                      load_data_chemprot_re(args, args.test_file, tokenizer)

    if args.analyze:
        output = load_file("analyze/output.json")
        t0,t1=[],[]
        for id, pred in output:
            instance = test_data[id]
            print("\nid:", id, "pred:", pred, " label:", instance["label"])

            t0.append(pred)
            t1.append(instance["label"])
            pp(" modal_data:")
            pp( instance["modal_data"])
        # for id, pred, tgt in output:
        #     instance = test_data[id]
        #     print("\nid:", id, " tgt:", tgt,  "pred:", pred, " label:", instance["label"])
        #
        #     t0.append(pred)
        #     t1.append(instance["label"])
        #     pp(" modal_data:")
        #     pp( instance["modal_data"])
        print(f1_score(t1, t0, average="micro"))
    else:
        # load model and data etc.
        args, model, optimizer = setup_common(args)

        # train_data, val_data, test_data = load_data(args.train_file), load_data(args.val_file), load_data(args.test_file)
        train(args, model, optimizer, (train_data, val_data, test_data))
