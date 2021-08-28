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
import torch.nn.utils.rnn as R
from typing import Tuple

class LstmFet(nn.Module):
    def __init__(self,args,
                 word_embed: nn.Embedding,
                 # lstm: nn.LSTM,
                 # output_linear: nn.Linear,
                 # word_embed_dropout: float = 0,
                 # lstm_dropout: float = 0,
                 # word_indices_for_labels=None,
                 # output_linear2: nn.Linear = None,
                 # label_embed=None,
                 # embed_dim=None,
                 label_num=None):
        super().__init__()

        self.word_embed = word_embed
        # self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.lstm = torch.nn.LSTM(args.embed_dim, args.embed_dim, batch_first=True,bidirectional=True)
        # self.output_linear = torch.nn.Linear(in_features=embed_dim * 2, out_features=args.out_dim)
        # self.output_linear2 = torch.nn.Linear(in_features=embed_dim * 2, out_features=embed_dim)
        # linear = torch.nn.Linear(embed_dim * 2, label_num)
        # linear2 = torch.nn.Linear(embed_dim * 2, embed_dim)
        # self.label_linear = torch.nn.Linear(embed_dim, embed_dim)
        # self.linear_to_attention_weight = torch.nn.Linear(embed_dim, 1)
        self.final_linear = torch.nn.Linear(in_features=args.embed_dim * 4, out_features=args.out_dim)
        self.loss = nn.MultiLabelSoftMarginLoss()

        self.word_embed_dropout = nn.Dropout(args.embed_dropout)
        self.lstm_dropout = nn.Dropout(args.lstm_dropout)
        # self.criterion = nn.MultiLabelSoftMarginLoss()
        # self.word_indices_for_labels = word_indices_for_labels
        # self.label_embed = label_embed

        # self.print_times_left = 1

    def forward(self,
                   batch,
                   # inputs: torch.Tensor,
                   # mention_masks: torch.Tensor,
                   # context_masks: torch.Tensor,
                   # seq_lens: torch.Tensor
                   args=None
                   ):
        """
        Args:
            inputs (torch.Tensor): Word index tensor for the input batch.
            mention_masks (torch.Tensor): A mention mask with the same size of
              `inputs`.
            context_masks (torch.Tensor): A context mask with the same size of
              `inputs`.
            seq_lens (torch.Tensor): A vector of sequence lengths.

            If a sequence has 6 tokens, where the 2nd token is a mention, and
            the longest sequence in the current batch has 8 tokens, the mention
            mask and context mask of this sequence are:
            - mention mask: [0, 1, 0, 0, 0, 0, 0, 0]
            - context mask: [1, 1, 1, 1, 1, 1, 0, 0]

        Returns:
            torch.Tensor: label scores. A NxM matrix where N is the batch size
              and M is the number of labels.
        """

        inputs=batch.word_ids
        mention_masks=batch.mention_masks
        context_masks=batch.context_masks
        seq_lens=batch.seq_lens
        in_train = batch.in_train
        labels = batch.labels



        inputs_embed = self.word_embed(inputs)
        inputs_embed = self.word_embed_dropout(inputs_embed)

        lstm_in = R.pack_padded_sequence(inputs_embed,
                                         seq_lens,
                                         batch_first=True, enforce_sorted=False)

        lstm_out = self.lstm(lstm_in)[0]
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
        # print("lstm_out", lstm_out.shape)
        lstm_out = self.lstm_dropout(lstm_out)

        # Average mention embedding
        num_mention_tokens=mention_masks.sum(1, keepdim=True)
        mention_masks = ((1 - mention_masks) * -1e14).softmax(-1).unsqueeze(-1)
        mention_repr = (lstm_out * mention_masks).sum(1)/num_mention_tokens
        # print("(lstm_out * mention_masks).sum(1)", (lstm_out * mention_masks).sum(1).shape)
        # print("mention_repr", mention_repr.shape)

        # Average context embedding
        num_context_tokens=context_masks.sum(1, keepdim=True)
        context_masks = ((1 - context_masks) * -1e14).softmax(-1).unsqueeze(-1)
        context_repr = (lstm_out * context_masks).sum(1)/num_context_tokens
        # print("context_repr", context_repr.shape)

        # Concatenate mention and context representations
        combine_repr = torch.cat([mention_repr, context_repr], dim=1)
        # print("combine_repr", combine_repr.shape)
        output = self.final_linear(combine_repr)
        # print("output", output.shape)


        if in_train:
            # label smoothing
            # return self.criterion(output, labels)
            return self.loss(output, labels)
            return torch.nn.functional.cross_entropy(output, labels)
            return torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)
        # print("sigmoid output")
        # pp(torch.sigmoid(output))
        pred_out = (torch.sigmoid(output) > 0.5).float()
        # print("pred_out")
        # pp(pred_out)
        # pred_out = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
        # print('pred_out', get_tensor_info(pred_out))
        return pred_out


        # # label attention to combined
        # label_embed_mat = self.word_embed_dropout(self.label_embed(self.word_indices_for_labels))
        # if self.print_times_left:
        #     print(label_embed_mat[:50])
        #     self.print_times_left -= 1
        # attention_weight = nn.functional.softmax(self.output_linear2(combine_repr).mm(label_embed_mat.t()), dim=-1)
        #
        # label_side_repr = attention_weight.mm(label_embed_mat)
        # # self_attention_weight=
        # final_repr = torch.cat([combine_repr, label_side_repr], dim=1)
        #
        # # # Linear classifier
        # # scores1 = self.output_linear(combine_repr)
        # # scores2 = self.output_linear2(combine_repr)
        #
        # # Linear classifier
        # scores1 = self.final_linear(final_repr)
        #
        # return scores1
# class LSTMEncoder(nn.Module):
#
#     def __init__(self, vocab_size=300, emb_size=300, hidden_size=300, num_layers=2, bidirectional=True,
#                  emb_p=0, input_p=0, hidden_p=0, output_p=0, pretrained_emb=None, pooling=True, pad=False, glove_weights=None):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.emb_size = emb_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.emb_p = emb_p
#         self.input_p = input_p
#         self.hidden_p = hidden_p
#         self.output_p = output_p
#         self.pooling = pooling
#
#         self.emb = EmbeddingDropout(nn.Embedding(vocab_size, emb_size), emb_p)
#         if pretrained_emb is not None:
#             self.emb.emb.weight.data.copy_(pretrained_emb)
#         else:
#             bias = np.sqrt(6.0 / emb_size)
#             nn.init.uniform_(self.emb.emb.weight, -bias, bias)
#         self.input_dropout = nn.Dropout(input_p)
#         self.output_dropout = nn.Dropout(output_p)
#         self.rnn = nn.LSTM(input_size=emb_size, hidden_size=(hidden_size // 2 if self.bidirectional else hidden_size),
#                            num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
#                            batch_first=True)
#         self.max_pool = MaxPoolLayer()
#
#     def forward(self, inputs, lengths):
#         """
#         inputs: tensor of shape (batch_size, seq_len)
#         lengths: tensor of shape (batch_size)
#
#         returns: tensor of shape (batch_size, hidden_size)
#         """
#         bz, full_length = inputs.size()
#         embed = self.emb(inputs)
#         embed = self.input_dropout(embed)
#         lstm_inputs = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
#         rnn_outputs, _ = self.rnn(lstm_inputs)
#         rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=True, total_length=full_length)
#         rnn_outputs = self.output_dropout(rnn_outputs)
#         return self.max_pool(rnn_outputs, lengths) if self.pooling else rnn_outputs
