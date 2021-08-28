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
from pprint import pprint as pp
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

from data import ChemetDataset

if __name__ == '__main__':
    args = read_args()

    # Specializing args for the experiment
    args.data_dir = "../data_online/chemet/"
    fname = "anno"
    args.train_file = args.data_dir + "distant_training_new.json"
    args.val_file = args.data_dir + "dev_anno_unseen_removed.json"
    args.test_file = args.data_dir + "test_anno_unseen_removed.json"

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
        "exp_id",
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

        "num_atom_types",
        "num_edge_types",
    ]
    args.downstream_layers = ["combiner", "gnn", "cm_attn", 'the_zero', 'the_one', 'rand_emb']

    # args.model_path=""
    # args.model_name = "fet_model"
    # args.model_name = "fet_model"
    args.exp = "fet"
    args.plm = "sci"
    args.plm = get_plm_fullname(args.plm)
    print("torch.cuda.device_count()", torch.cuda.device_count())
    # if torch.cuda.device_count() > 2:
    #     args.gpu_id = 2

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
    #### for LSTM input
    word_embed=None
    if args.model_name=="lstm":

        main_dir="/"
        word_embed_type = "glove.840B.300d"
        args.embed_dim=300
        word_embed_type = "patent_w2v"
        args.embed_dim=200
        embed_file = os.path.join('../embeddings/' + word_embed_type + '.txt')
        word_embed_path = os.path.join("../embeddings", word_embed_type + "word_embed.pkl")
        word_vocab_path = os.path.join("../embeddings", word_embed_type + "word_vocab.pkl")
        files_vocab_path = None
        if not (os.path.isfile(word_embed_path)) or not (os.path.isfile(word_vocab_path)):
            print("No word_embed or word_vocab save, dumping...")
            word_embed, word_vocab = load_word_embed(embed_file,
                                                     args.embed_dim,
                                                     skip_first=True, file_vocab_path=files_vocab_path)
            pkl.dump(word_embed, open(word_embed_path, "wb"))
            pkl.dump(word_vocab, open(word_vocab_path, "wb"))
            print("word_embed Saved")
        word_embed = pkl.load(open(word_embed_path, "rb"))
        word_vocab = pkl.load(open(word_vocab_path, "rb"))
        # reversed_word_vocab = {value: key for (key, value) in word_vocab.items()}
        # vocabs = {'word': word_vocab}
        for data_split in [train_data, val_data, test_data]:
            for i, sample in enumerate(data_split):
                sample["word_ids"]=[word_vocab.get(t, word_vocab.get(t.lower(), 0))
                          for t in sample["original_text"]]
                sample["mention_masks"]=[1 if sample["original_pos"][0] <= i < sample["original_pos"][1] else 0
                                for i in range(len(sample["original_text"]))]
                sample["context_masks"]=[1 for _ in range(len(sample["original_text"]))]
                sample["is_rnn"]=True
    else:
        for data_split in [train_data, val_data, test_data]:
            for i, sample in enumerate(data_split):
                sample["is_rnn"]=False

    #
    # # add glove indicies
    # def load_glove_vectors(glove_file="./data/glove.6B/glove.6B.50d.txt"):
    #     """Load the glove word vectors"""
    #     word_vectors = {}
    #     with open(glove_file) as f:
    #         for line in f:
    #             split = line.split()
    #             word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
    #     return word_vectors
    #
    #
    # def get_emb_matrix(pretrained, word_counts, emb_size=50):
    #     """ Creates embedding matrix from word vectors"""
    #     vocab_size = len(word_counts) + 2
    #     vocab_to_idx = {}
    #     vocab = ["", "UNK"]
    #     W = np.zeros((vocab_size, emb_size), dtype="float32")
    #     W[0] = np.zeros(emb_size, dtype='float32')  # adding a vector for padding
    #     W[1] = np.random.uniform(-0.25, 0.25, emb_size)  # adding a vector for unknown words
    #     vocab_to_idx["UNK"] = 1
    #     i = 2
    #     for word in word_counts:
    #         if word in word_vecs:
    #             W[i] = word_vecs[word]
    #         else:
    #             W[i] = np.random.uniform(-0.25, 0.25, emb_size)
    #         vocab_to_idx[word] = i
    #         vocab.append(word)
    #         i += 1
    #     return W, np.array(vocab), vocab_to_idx
    #
    #
    # word_vecs = load_glove_vectors()
    # pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, counts)


    # exit()
    print("args.num_atom_types,args.num_edge_types", args.num_atom_types, args.num_edge_types)
    args, model, optimizer = setup_common(args, word_embed)

    # train or analyze
    if not args.eval:
        train(args, model, optimizer, (train_data, val_data, test_data))

    else:
        print("Eval")

        model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
        test_score, output = evaluate(args, model, test_data)
        print(test_score)
        # val_score, output2 = evaluate(args, model, val_data)
        # print(val_score)

        if args.error_analysis:
            sample_id = 0



            original_data = load_file(test_file)

            final_data = []
            for idx in range(len(original_data)):
                sample = original_data[idx]

                text = sample["tokens"]
                for mention in sample["annotations"]:
                    sample_id += 1
                    m_s, m_e = mention["start"], mention["end"]
                    m = " ".join(text[m_s:m_e])
                    m = m.replace("  ", " ")
                    final_data.append({"original_text": text, 'mention_name': m, "original_labels": mention["labels"]})

            for id, pred, label in output:
                print("\n\nsample", id)
                sample = final_data[id]
                # print("text is", sample["original_text"])
                print(" ".join(sample["original_text"]) )
                print("\nmention is", sample['mention_name'])
                print("original labels are")
                pp(sorted(sample["original_labels"]))

                here_labels = sorted([labels[i] for i, c in enumerate(pred) if label[i] == 1])
                predicted_labels = sorted([labels[i] for i, c in enumerate(pred) if c == 1])

                # print("has labels", sorted(here_labels))
                print("predicted labels")
                pp(sorted(predicted_labels))

                here_labels = set(here_labels)
                predicted_labels = set(predicted_labels)

                missed_labels = here_labels.difference(predicted_labels)
                incorrected_included_labels = predicted_labels.difference(here_labels)
                if missed_labels:
                    print("missed_labels")
                    pp(missed_labels)
                if incorrected_included_labels:
                    print("incorrected_included_labels")
                    pp(incorrected_included_labels)
        # if args.attn_analysis:

        # for i, c in enumerate(pred):
        #     if label[i] == 1:
        #         print("has label", labels[i])
        # for i, c in enumerate(pred):
        #     if c == 1:
        #         print("predicted label", labels[i])

        # if label[i] != c:
        #     print("pred")
        #     print(labels[i])

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
