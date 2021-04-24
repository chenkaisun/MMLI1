from train import train
from data import load_data_chemprot_re, ChemProtDataset, ModalRetriever
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
from evaluate import evaluate


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    args = read_args()
    data_dir = "data_online/ChemProt_Corpus/chemprot_preprocessed/"
    train_file = data_dir + "train.txt"
    val_file = data_dir + "dev.txt"
    test_file = data_dir + "test.txt"
    # if args.debug:
        # args.plm="bert-ba"
        # args.use_amp=False
        # args.use_cache=False
        # args.use_cache=True
        # data
    args.model_name = "re_model"
    args.exp = "re"
    plm_dict={
        "base":"bert-base-cased",
        "sap":"cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        "sci":"allenai/scibert_scivocab_uncased"
    }
    args.plm=plm_dict[args.plm]
    # args.plm = "allenai/scibert_scivocab_uncased"
    # args.plm = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    # if args.model_type=="tg": args.g_global_pooling=1
    # print("args.g_global_pooling first",args.g_global_pooling)
    if args.debug:
        args.plm = "prajjwal1/bert-tiny"
        # args.num_epoch = 50
        args.batch_size = 2
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
    print("args.plm",args.plm)
    set_seeds(args)
    print("tokenizer1")
    tokenizer = get_tokenizer(args.plm)
    print("tokenizer2")

    modal_retriever = ModalRetriever()
    train_data, val_data, test_data = ChemProtDataset(args, train_file, tokenizer, modal_retriever), \
                                      ChemProtDataset(args, val_file, tokenizer, modal_retriever), \
                                      ChemProtDataset(args, test_file, tokenizer, modal_retriever)

    # ef


    if args.analyze:
        args, model, optimizer = setup_common(args)

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
        t0,t1=[],[]
        null_cnt=0
        total_cnt=0
        for id, pred in output:
            instance = test_data[id]
            if instance["label"]!=pred:
                total_cnt+=1
                if 0 in [instance["modal_data"][0][2], instance["modal_data"][0][3], instance["modal_data"][1][1]]: null_cnt+=1

                print("\nid:", id, "pred:", rels[pred], " label:", rels[instance["label"]])
                print(str(instance["text"].encode(errors="ignore")))
                t0.append(pred)
                t1.append(instance["label"])
                print("modal_data:")
                pp( instance["modal_data"])
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
    else:
        # load model and data etc.
        args, model, optimizer = setup_common(args)

        # train_data, val_data, test_data = load_data(args.train_file), load_data(args.val_file), load_data(args.test_file)
        train(args, model, optimizer, (train_data, val_data, test_data))
