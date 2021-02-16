from pprint import pprint as pp
import pickle as pkl
from collections import deque, defaultdict, Counter
import json
from IPython import embed
import random
import time
import os
import numpy as np
import argparse
import gc
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, degree
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch.nn import Dropout, GELU
from torch_geometric.nn import global_mean_pool, global_max_pool, BatchNorm
from torch.nn import Linear
from transformers import BertModel, AutoModel

from model.model_utils import get_tensor_info

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # a=Linear(5, 32)
        # b=Linear(32, 32)
        # c=Linear(32, 32)
        # d=Linear(32, 32)
        # e=Linear(32, 32)
        self.dropout=args.dropout
        # self.ls=torch.nn.Sequential(
        #     Dropout(self.dropout),
        #     GINConv(Linear(5, 32)),
        #     GELU(),
        #     Dropout(self.dropout),
        #     BatchNorm(32),
        #     GINConv(Linear(32, 32)),
        #     GELU(),
        #     Dropout(self.dropout),
        #     BatchNorm(32),
        #     GINConv(Linear(32, 32)),
        # )
        self.mid=torch.nn.Sequential(
            # GELU(),
            # Dropout(self.dropout),
            BatchNorm(32),
        )
        # self.conv1 = GINConv(nn=Linear(5, 32))
        # self.conv2 = GINConv(nn=Linear(32, 32))
        # self.conv3 = GINConv(nn=Linear(32, 32))
        # self.conv4 = GINConv(nn=Linear(32, 32))
        # self.conv5 = GINConv(nn=Linear(32, 32))
        self.conv1 = GATConv(5, 32)
        self.conv2 = GATConv(32, 32)
        self.conv3 = GATConv(32, 32)
        self.conv4 = GATConv(32, 32)
        self.conv5 = GATConv(32, 32)
        # self.conv1 = GATConv(5, 32)
        # self.conv2 = GATConv(32, 32)
        # self.conv3 = GATConv(32, 32)
        # self.conv4 = GATConv(32, 32)
        # self.conv5 = GATConv(32, 32)
        # self.conv3 = GATConv(128, 256)
        # self.lin = Linear(16, 2)

    def forward(self, data):
        # num_attr, list_num_atoms = outcome.get_network_params()
        # train, val, test = outcome.splitter(list(outcome.yielder()))
        edge_index = data.edge_index
        x = data.x
        batch = data.batch

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)
        #
        # # Step 3: Compute normalization.
        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        #
        # # Step 4-5: Start propagating messages.
        # return self.propagate(edge_index, x=x, norm=norm)

        # print(get_tensor_info(x))
        # print(get_tensor_info(edge_index))

        # x = self.conv1(x, edge_index)
        # x = x.relu()
        # for layer in [self.conv1]:#, self.conv4, self.conv5 , self.conv2, self.conv3
        #     x = layer(x, edge_index)
        #     x = x.relu()

        # print("x before", x)
        # print("x", x)
        # x=self.dropout(x)
        # x=torch.dropout(x, p=self.dropout, train=self.training)
        x=self.conv1(x, edge_index)
        # x=self.mid(x)
        # torch.nn.functional.tanh(x)
        # x=self.mid(x)
        x=self.conv2(x, edge_index)
        # torch.nn.functional.tanh(x)
        # x=self.mid(x)
        # x=self.mid(x)
        x=self.conv3(x, edge_index)

        # x = x.relu()
        # print("x", x)
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)
        # x = self.conv1(x.float(), edge_index.long())
        # x = F.relu(x)
        # x = F.dropout(x, p=self.dropout,training=self.training)
        # x = self.conv2(x.float(), edge_index.long())
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.lin(x)

        return x
        return F.log_softmax(x, dim=1)



class JNet(torch.nn.Module):
    def __init__(self, args):
        super(JNet, self).__init__()
        self.gnn = Net(args)
        self.plm = AutoModel.from_pretrained(args.plm)
        # self.combiner = Linear(16+128, 1)
        # self.combiner = Linear(32, 1)
        # self.combiner = Linear(args.plm_hidden_dim+32, 1)
        self.combiner = Linear(args.plm_hidden_dim, 1)
        self.dropout=args.dropout

    def forward(self, input):
        input_ids=input['input_ids']
        batch_graph_data=input['batch_graph_data']
        ids=input['ids']
        in_train=input['in_train']
        g_out=self.gnn(batch_graph_data)
        t_out=self.plm(**input_ids, return_dict=True).pooler_output
        # print(get_tensor_info(g_out))
        # print(get_tensor_info(t_out))
        # print("g_out", get_tensor_info(g_out))
        # print("t_out", get_tensor_info(t_out))

        # output=self.combiner(torch.cat([g_out, t_out], dim=-1))
        output=self.combiner(torch.cat([t_out], dim=-1))
        # output=self.combiner(torch.cat([g_out], dim=-1))
        # print("output",output.shape)
        # print("batch_graph_data.y.unsqueeze(-1)",batch_graph_data.y.shape)
        if in_train:
            return torch.nn.functional.binary_cross_entropy_with_logits(output, batch_graph_data.y.view(-1, 1))
        # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)
        return torch.sigmoid(output)



        return F.log_softmax(x, dim=1)
