import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
from torch.nn import Linear
from transformers import AutoModel

from model.gnn import MoleGNN, MoleGNN2
# from train_utils import get_tensor_info
from model.model_utils import get_tensor_info, LabelSmoothingCrossEntropy, CrossModalAttention
import numpy as np
from IPython import embed


# from torchtext.vocab import GloVe

class NER(torch.nn.Module):
    def __init__(self, args):
        super(NER, self).__init__()

        self.model_type = args.model_type

        if 'g' in args.model_type:
            # self.gnn = MoleGNN(args)
            self.gnn = MoleGNN2(args)

        """===================================="""
        """Try 2 Berts"""
        """===================================="""
        self.plm = AutoModel.from_pretrained(args.plm)
        # if args.g_only:
        #     self.combiner = Linear(args.g_dim, 1)
        # elif args.t_only:
        #     self.combiner = Linear(args.plm_hidden_dim, 1)
        # else:
        # self.combiner = Linear(args.g_dim, 1)
        # self.combiner = Linear(args.plm_hidden_dim, 1)

        final_dim = args.plm_hidden_dim
        if 'tdgx' in args.model_type:
            print("tdgx")

            # two ent plm*2, mol modal 800, prot modal plm
            # self.combiner = Linear(args.plm_hidden_dim * 3 + final_dim, args.out_dim)
            self.combiner = Linear(args.plm_hidden_dim * 5, args.out_dim)
        elif 'tdg' in args.model_type:
            print("args.tdg")
            # self.combiner = Linear(args.plm_hidden_dim * 3 + final_dim, args.out_dim)
            self.combiner = Linear(args.plm_hidden_dim * 5, args.out_dim)
            # self.combiner = Linear(args.plm_hidden_dim * 4, args.out_dim)

            # args.plm_hidden_dim
        elif 'tg' in args.model_type:
            print("args.tg")
            self.combiner = Linear(args.plm_hidden_dim * 3 + args.g_dim, args.out_dim)
        elif 'td' in args.model_type:
            print("args.td")
            self.combiner = Linear(args.plm_hidden_dim * 4, args.out_dim)
        elif 't' in args.model_type:
            print("args.t")
            # self.combiner = Linear(args.plm_hidden_dim, args.out_dim)
            # self.combiner = Linear(args.plm_hidden_dim*3, args.out_dim)
            self.combiner = Linear(args.plm_hidden_dim * 3, args.out_dim)
            # self.combiner = Linear(args.plm_hidden_dim, args.out_dim)
            if args.add_label_text:
                print("add_label_text")
                self.combiner = Linear(args.plm_hidden_dim * 4, args.out_dim)

        # self.map2smaller = Linear(args.plm_hidden_dim, args.g_dim)
        self.text_transform = Linear(args.plm_hidden_dim, args.g_dim)

        self.criterion = LabelSmoothingCrossEntropy(reduction='sum')
        # self.dropout = torch.nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.loss = torch.nn.CrossEntropyLoss()
        self.cm_attn = CrossModalAttention(reduction='mean', m1_dim=args.g_dim, m2_dim=args.plm_hidden_dim,
                                           final_dim=final_dim)
        self.emb = nn.Embedding(1, args.plm_hidden_dim)
        self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device)
        self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)

    def forward(self, input, args):
        pass
