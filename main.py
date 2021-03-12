from train import train
from data import load_data_chemprot_re ,load_mole_data
from train_utils import setup_common
from utils import mkdir
from model.load_model import *
from transformers import BertTokenizer
from transformers import AutoTokenizer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

from train_utils import *

if __name__ == '__main__':
    args = read_args()
    if args.debug:
        # args.plm="bert-ba"
        # args.use_amp=False
        args.use_cache = True
        args.use_cache = False
        # args.train_file="data/property_pred/BBBP.csv"
        # args.tgt_name="p_np"
        # args.train_file="cmpd_info.json"
        args.train_file = "data/property_pred/clintox.csv"
        args.tgt_name = 'CT_TOX'
        args.num_epoch = 100
        args.batch_size = 32
        args.burn_in = 1
        args.lr = 1e-4
        args.grad_accumulation_steps = 1
        args.plm = "prajjwal1/bert-tiny"
        args.patience = 5
        args.g_dim = 128
        # args.g_only = True
        args.exp = "mol_pred"
        args.model_name="joint_gt"
        # args.t_only=True
        # args.g_dim=32
        # args.use_cache=False


    # Trainer class?
    set_seeds(args)
    print("tokenizer1")
    tokenizer = AutoTokenizer.from_pretrained(args.plm, cache_dir="/")
    print("tokenizer2")

    train_data, val_data, test_data = load_mole_data(args, args.train_file, tokenizer)

    # load model and data etc.
    args, model, optimizer = setup_common(args)

    # train_data, val_data, test_data = load_data(args.train_file), load_data(args.val_file), load_data(args.test_file)
    train(args, model, optimizer, (train_data, val_data, test_data))
