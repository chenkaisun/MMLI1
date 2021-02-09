from train import train
from data import load_data
from utils import setup_common
from model.load_model import *
from transformers import BertTokenizer
from utils import mkdir
from transformers import AutoTokenizer
torch.backends.cudnn.benchmark=True

if __name__ == '__main__':
    # load model and data etc.
    args, model, optimizer = setup_common()
    mkdir("model")
    mkdir("model/states")
    tokenizer=AutoTokenizer.from_pretrained(args.plm)

    train_data, val_data, test_data = load_data(args, args.train_file, tokenizer)
    # train_data, val_data, test_data = load_data(args.train_file), load_data(args.val_file), load_data(args.test_file)
    train(args, model, optimizer, (train_data, val_data, test_data))
