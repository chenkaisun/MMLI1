from train import train
from data import load_data
from train_utils import setup_common
from utils import mkdir
from model.load_model import *
from transformers import BertTokenizer
from transformers import AutoTokenizer
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic = True

from train_utils import *

if __name__ == '__main__':
    args = read_args()
    if args.debug:
        # args.plm="bert-ba"
        # args.use_amp=False
        args.use_cache=True
        # data
        data_dir="data_online/ChemProt_Corpus/chemprot_preprocessed/"
        args.train_file=data_dir+"train.txt"
        args.val_file=data_dir+"dev.txt"
        args.test_file=data_dir+"test.txt"
        args.exp="re"

        args.tgt_name='CT_TOX'
        args.num_epoch = 100
        args.batch_size = 24
        args.burn_in = 1
        args.lr=1e-4
        args.grad_accumulation_steps = 1
        args.plm="prajjwal1/bert-tiny"
        args.patience=5
        args.g_dim=64
        args.g_only=True
        # args.t_only=True
        # args.g_dim=32
        # args.use_cache=False

    set_seeds(args)
    print("tokenizer1")
    tokenizer=AutoTokenizer.from_pretrained(args.plm, cache_dir="/")
    print("tokenizer2")
    train_data, val_data, test_data = load_data(args, args.train_file, tokenizer)

    # load model and data etc.
    args, model, optimizer = setup_common(args)

    # train_data, val_data, test_data = load_data(args.train_file), load_data(args.val_file), load_data(args.test_file)
    train(args, model, optimizer, (train_data, val_data, test_data))
