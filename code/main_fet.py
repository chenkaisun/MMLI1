# from rdkit import Chem
# from torch_geometric.data import Data, Batch

from train import train
from data import ModalRetriever
# import torch
import os
# from train_utils import setup_common
# from utils import mkdir
# from model.load_model import get
# from transformers import BertTokenizer
# from transformers import AutoTokenizer
from options import read_args
from train_utils import *

from utils import dump_file, load_file
from pprint import pprint as pp
from sklearn.metrics import f1_score
from evaluate import evaluate

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

from data import ChemetDataset

if __name__ == '__main__':
    args = read_args()

    # Specializing args for the experiment
    args.data_dir = "../data_online/chemet/"
    fname = "test_jinfeng_b"
    args.train_file = args.data_dir + "test_jinfeng_b_cleaned_cleaned.json"
    args.val_file = args.data_dir + "test_chem_anno_cleaned_cleaned.json"
    args.test_file = args.data_dir + "test_chem_anno_cleaned_cleaned.json"

    data_dir = args.data_dir
    train_file = args.train_file
    test_file = args.test_file
    val_file = args.val_file
    # args.val_file = data_dir + "dev.txt"
    # args.test_file = data_dir + "test.txt"

    # set params Fixed + Tunable
    args.useful_params = [
        # fixed
        "exp",
        "max_grad_norm",
        "g_global_pooling",
        "mult_mask",
        "g_mult_mask",
        "grad_accumulation_steps",
        "model_name",

        # tuning
        "activation",
        "batch_size",
        "cm_type",
        "debug",
        "dropout",
        "gnn_type",
        "g_dim",
        "patience",
        "plm",
        "plm_hidden_dim",
        "plm_lr",
        "pool_type",
        "lr",
        "model_type",
        "num_epochs",
        "num_gnn_layers",
        "use_cache",
    ]
    args.downstream_layers = ["combiner", "gnn", "cm_attn", 'the_zero', 'the_one', 'rand_emb']

    args.model_name = "fet_model"
    args.exp = "fet"
    args.plm = "sci"
    args.plm = get_plm_fullname(args.plm)
    if torch.cuda.device_count() > 2:
        args.gpu_id = 2

    # args.num_epoch = 50
    # args.batch_size = 8
    # args.g_dim = 128
    # args.patience = 8
    # args.lr = 1e-4
    # args.plm_lr = 3e-5
    # args.use_cache = True
    # args.model_type = "tdg"
    # args.activation = "gelu"

    if args.debug:
        print("Debug Mode ON")
        args.plm = get_plm_fullname("tiny")
        args.batch_size = 2
        args.num_epochs = 2
        args.patience = 3
        # args.use_cache = False
    print("PLM is", args.plm)
    print("model is", args.model_name)

    # Prepare Data and Model

    set_seeds(args)
    tokenizer = get_tokenizer(args.plm)

    modal_retriever = ModalRetriever(data_dir + "mention2ent.json", data_dir + "cmpd_info.json")

    labels_path = data_dir + fname + "_labels.json"
    if not args.use_cache or not os.path.exists(labels_path):
        labels = ChemetDataset.collect_labels([train_file, val_file, test_file], labels_path)
    else:
        labels = load_file(labels_path)
    # print("labels", labels)

    train_data, val_data, test_data = ChemetDataset(args, train_file, tokenizer, modal_retriever, labels), \
                                      ChemetDataset(args, val_file, tokenizer, modal_retriever, labels), \
                                      ChemetDataset(args, test_file, tokenizer, modal_retriever, labels)

    args, model, optimizer = setup_common(args)

    # train or analyze
    if not args.eval:
        train(args, model, optimizer, (train_data, val_data, test_data))

    else:
        print("Eval")

        model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
        test_score, output = evaluate(args, model, test_data)
        print(test_score)
        exit()
        rels = ['AGONIST-ACTIVATOR',
                'DOWNREGULATOR', 'SUBSTRATE_PRODUCT-OF',
                'AGONIST', 'INHIBITOR',
                'PRODUCT-OF', 'ANTAGONIST',
                'ACTIVATOR', 'INDIRECT-UPREGULATOR',
                'SUBSTRATE', 'INDIRECT-DOWNREGULATOR',
                'AGONIST-INHIBITOR', 'UPREGULATOR', ]

        output = load_file("analyze/output.json")
        t0, t1 = [], []
        null_cnt = 0
        total_cnt = 0
        for id, pred in output:
            instance = test_data[id]
            if instance["label"] != pred:
                total_cnt += 1
                if 0 in [instance["modal_data"][0][2], instance["modal_data"][0][3],
                         instance["modal_data"][1][1]]: null_cnt += 1

                print("\nid:", id, "pred:", rels[pred], " label:", rels[instance["label"]])
                print(str(instance["text"].encode(errors="ignore")))
                t0.append(pred)
                t1.append(instance["label"])
                print("modal_data:")
                pp(instance["modal_data"])
        # for id, pred, tgt in output:
        #     instance = test_data[id]
        #     print("\nid:", id, " tgt:", tgt,  "pred:", pred, " label:", instance["label"])
        #
        #     t0.append(pred)
        #     t1.append(instance["label"])
        #     pp(" modal_data:")
        #     pp( instance["modal_data"])
        print(null_cnt)
        print(total_cnt)
        print(f1_score(t1, t0, average="micro"))
