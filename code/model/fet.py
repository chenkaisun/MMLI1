import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
from torch.nn import Linear
from transformers import AutoModel

from model.gnn import *
# from train_utils import get_tensor_info
from model.model_utils import get_tensor_info, LabelSmoothingCrossEntropy
import numpy as np
from IPython import embed
from torch.nn import TransformerEncoderLayer

from utils import get_gpu_mem_info
from pprint import pprint as pp
# from torchtext.vocab import GloVe
class CrossModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.num_gnn_layers = args.num_gnn_layers
        self.cm_type = args.cm_type
        self.pool_type = args.pool_type
        # self.reduction = reduction
        # self.temprature = 3
        # self.aggregate = True
        # self.l_filter1 = torch.nn.Linear(m1_dim, final_dim)
        # self.l_filter2 = torch.nn.Linear(m2_dim, final_dim)
        #
        # self.l1 = torch.nn.Linear(m1_dim, final_dim)
        # self.l2 = torch.nn.Linear(m2_dim, final_dim)

        if args.cm_type == 0:
            self.transformers = ModuleList()
            for _ in range(args.num_gnn_layers):
                self.transformers.append(TransformerEncoderLayer(d_model=args.plm_hidden_dim, nhead=8))

        self.gnns = ModuleList()
        self.gnns.append(MoleGraphConv(args, is_first_layer=True))
        for _ in range(args.num_gnn_layers - 1):
            self.gnns.append(MoleGraphConv(args))

        self.ls = ModuleList()
        for _ in range(args.num_gnn_layers):
            self.ls.append(Linear(args.plm_hidden_dim, args.plm_hidden_dim))

        self.atom_lin = Linear(args.plm_hidden_dim, args.plm_hidden_dim)

        # self.rand_g_emb = nn.Embedding(1, args.plm_hidden_dim)
        # self.rand_d_emb = nn.Embedding(1, args.plm_hidden_dim)
        # self.rand_gd_emb = nn.Embedding(1, args.plm_hidden_dim)
        self.rand_emb = nn.Embedding(1, args.plm_hidden_dim)

        self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device)
        self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)

    def forward(self, graph, text, batch_ent1_d_mask, batch_ent1_g_mask):
        """
        :param graph: graph data
        :param text: description
        :return: attended embeddings
        """
        # print("\nCM fwd")

        x = graph.x
        # batch_num_nodes = x.shape[0]
        # graph_ids = torch.unique(graph.batch).cpu().numpy().astype(int)
        # print("graph", get_tensor_info(x))

        # text = torch.nn.functional.gelu(self.l1(text))
        # print("text", get_tensor_info(text))

        cms = []

        for i in range(self.num_gnn_layers):
            x = self.gnns[i](x, graph)
            tmp_x = x
            # tmp_x = torch.tanh(self.ls[i](x))

            """split since graphs have different num nodes"""

            attended = []
            for batch_id in range(text.shape[0]):
                # print("batch_id", batch_id)

                g_indices = [i for i, bl in enumerate(graph.batch == batch_id) if bl]
                # print("graph.batch", get_tensor_info(graph.batch))
                # print("g_indices", g_indices)

                x_i = torch.index_select(tmp_x, 0, torch.tensor(g_indices, dtype=torch.long, device=x.device))
                t_i = text[batch_id]

                if self.cm_type == 0:

                    stacked = torch.cat([x_i, t_i], dim=0).unsqueeze(0)
                    # print("stacked", get_tensor_info(stacked))
                    # attended = self.transformers[i](stacked).squeeze(0)
                    attended.append(self.transformers[i](stacked).squeeze(0))
                else:
                    g2t_sim = x_i.mm(t_i.t())
                    t2g_sim = t_i.mm(x_i.t())
                    g2t_sim=torch.softmax(g2t_sim, dim=-1)
                    t2g_sim=torch.softmax(t2g_sim, dim=-1)
                    attended.append(torch.cat([g2t_sim.mm(t_i), t2g_sim.mm(x_i)], dim=0))

                if self.pool_type in [0, 2]:
                    attended[-1] = torch.mean(attended[-1], dim=0, keepdim=True)
                else:
                    attended[-1] = torch.max(attended[-1], dim=0, keepdim=True)[0]
                # print("attended[-1]", get_tensor_info(attended[-1]))

            attended = torch.stack(attended, dim=0)
            # print("attended", get_tensor_info(attended))
            # cms.append(torch.mean(attended, dim=0))
            cms.append(attended)
            # g_indices=[i for i, bl in enumerate(batch_ent1_g.batch == graph_id) if bl]

        out_t = text[:, 0, :]
        # print("out_t.shape", get_tensor_info(out_t))

        # out_g = torch.mean(x, dim=0)
        x = scatter(x, graph.batch, dim=0, reduce='mean')
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_lin(x)
        out_g = torch.tanh(x)
        # print("out_g.shape", get_tensor_info(out_t))

        if self.pool_type in [1, 2]:
            out_cms = torch.mean(torch.cat(cms, dim=1), dim=1)
        else:
            out_cms = torch.max(torch.cat(cms, dim=1), dim=1)[0]
        # print("out_cms.shape", get_tensor_info(out_cms))

        # batch_ent1_d_mask=torch.tensor([[1],[0]])self.emb(self.the_zero)
        batch_ent1_dg_mask = batch_ent1_d_mask * batch_ent1_g_mask
        return torch.cat([out_t * batch_ent1_d_mask + (1 - batch_ent1_d_mask) * self.rand_emb(self.the_zero),
                          out_g * batch_ent1_g_mask + (1 - batch_ent1_g_mask) * self.rand_emb(self.the_zero),
                          out_cms * batch_ent1_dg_mask + (1 - batch_ent1_dg_mask) * self.rand_emb(self.the_zero)],
                         dim=-1)

        return attended[:graph.shape[0]], attended[graph.shape[0]:]

        # stacked = torch.cat([graph, text], dim=0).unsqueeze(0)
        # print("stacked", get_tensor_info(stacked))
        # attended = self.transformer(torch.cat([graph, text], dim=0).unsqueeze(0)).squeeze(0)
        # print("attended", get_tensor_info(attended))

        return attended[:graph.shape[0]], attended[graph.shape[0]:]


class FET(torch.nn.Module):
    def __init__(self, args):
        super(FET, self).__init__()

        self.model_type = args.model_type
        self.plm = AutoModel.from_pretrained(args.plm)

        if "g" in self.model_type and "m" not in self.model_type:
            self.gnn = MoleGNN2(args)
        if "m" in self.model_type:
            self.cm_attn = CrossModal(args)
        final_dim = args.plm_hidden_dim

        if 'tdgm' in args.model_type:
            print("args.tdgm")
            self.combiner = Linear(args.plm_hidden_dim * 5, args.out_dim)
        elif 'dgm' in args.model_type:
            print("args.dgm")
            self.combiner = Linear(args.plm_hidden_dim * 4, args.out_dim)
        elif 'tdg' in args.model_type:
            print("args.tdg")
            self.combiner = Linear(args.plm_hidden_dim * 4, args.out_dim)
        elif 'tg' in args.model_type or 'td' in args.model_type:
            print("args.tg or td")
            self.combiner = Linear(args.plm_hidden_dim * 3, args.out_dim)
        elif 'd' in args.model_type or 'g' in args.model_type or 't' in args.model_type:
            print("args.d or g or t")
            self.combiner = Linear(args.plm_hidden_dim * 2, args.out_dim)
        elif 's' in args.model_type:
            print("args.s")
            self.combiner = Linear(args.plm_hidden_dim, args.out_dim)
            # if args.add_label_text:
            #     print("add_label_text")
            #     self.combiner = Linear(args.plm_hidden_dim * 4, args.out_dim)

        # self.text_transform = Linear(args.plm_hidden_dim, args.g_dim)

        # self.criterion = LabelSmoothingCrossEntropy(reduction='sum')
        self.dropout = args.dropout

        # self.loss = torch.nn.CrossEntropyLoss()
        self.loss = nn.MultiLabelSoftMarginLoss()

        if ("d" in self.model_type or "g" in self.model_type) and "m" not in self.model_type:
            self.rand_emb = nn.Embedding(2, args.plm_hidden_dim)

        self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device)
        self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)

        # self.emb = nn.Embedding(1, args.plm_hidden_dim)
        # self.rand_g_emb = nn.Embedding(1, args.plm_hidden_dim)
        # self.rand_d_emb = nn.Embedding(1, args.plm_hidden_dim)
        # self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device)
        # self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)

    def forward(self, input, args):
        texts = input.texts
        texts_attn_mask = input.texts_attn_mask

        masked_texts = input.masked_texts
        masked_texts_attn_mask = input.masked_texts_attn_mask

        labels = input.labels
        # xi[xi != xi] = 0
        batch_ent1_d = input.ent1_d
        batch_ent1_d_mask = input.ent1_d_mask
        batch_ent1_g = input.ent1_g
        batch_ent1_g_mask = input.ent1_g_mask

        ent1_pos = input.ent1_pos
        ent1_masked_pos = input.ent1_masked_pos

        # concepts = input.concepts
        in_train = input.in_train

        # print("texts",get_tensor_info(texts))
        # print("texts_attn_mask",get_tensor_info(texts_attn_mask))
        # print("masked_texts",get_tensor_info(masked_texts))
        # print("masked_texts_attn_mask",get_tensor_info(masked_texts_attn_mask))
        # # print("batch_ent1_d",get_tensor_info(batch_ent1_d))
        # print("batch_ent1_d_mask",get_tensor_info(batch_ent1_d_mask))
        # # print("batch_ent1_g",get_tensor_info(batch_ent1_g))
        # print("batch_ent1_g_mask",get_tensor_info(batch_ent1_g_mask))
        # print("ent1_pos",get_tensor_info(ent1_pos), ent1_pos)
        # print("labels",get_tensor_info(labels))

        # label_text = input.label_text

        # print("batch_graph_data", batch_graph_data)

        # bert
        # token_id_offset = 1
        # modals = []
        hid_texts, hid_ent1_d, hid_ent1_g = None, None, None
        hid_label_text = None

        final_vec = []

        "=========Original Text Encoder=========="
        # embed()
        # print("texts", get_tensor_info(texts))
        # print("texts_attn_mask", get_tensor_info(texts_attn_mask))
        hid_texts = self.plm(input_ids=texts, attention_mask=texts_attn_mask, return_dict=True).last_hidden_state

        hid_mask_texts = None
        if "t" in self.model_type:
            hid_mask_texts = self.plm(input_ids=masked_texts, attention_mask=masked_texts_attn_mask,
                                      return_dict=True).last_hidden_state

        # print("hid_texts",get_tensor_info(hid_texts))
        # print("hid_mask_texts",get_tensor_info(hid_mask_texts))
        ent1_embeds = []
        ent1_mask_embeds = []
        for i in range(hid_texts.shape[0]):
            text_embed = hid_texts[i]
            ent1_embeds.append(text_embed[ent1_pos[i, 0]])
            if "t" in self.model_type:
                mask_text_embed = hid_mask_texts[i]
                ent1_mask_embeds.append(mask_text_embed[ent1_masked_pos[i, 0]])
            # print("text_embed[ent1_pos[i, 0]]", get_tensor_info(text_embed[ent1_pos[i, 0]]))
        ent1_embeds = torch.stack(ent1_embeds, dim=0)  # [n_e, d]
        final_vec.append(ent1_embeds)

        if "t" in self.model_type:
            ent1_mask_embeds = torch.stack(ent1_mask_embeds, dim=0)  # [n_e, d]
            final_vec.append(ent1_mask_embeds)

        "=========Multi-Modal=========="

        if "d" in self.model_type:
            hid_ent1_d = self.plm(**batch_ent1_d, return_dict=True).last_hidden_state
            if "m" not in self.model_type:
                final_vec.append(
                    hid_ent1_d[:, 0, :] * batch_ent1_d_mask + (1 - batch_ent1_d_mask) * self.rand_emb(self.the_zero))
        # print("0")
        # get_gpu_mem_info()
        if "g" in self.model_type and "m" not in self.model_type:
            hid_ent1_g = self.gnn(batch_ent1_g, args.g_global_pooling)  # * batch_ent1_g_mask
            final_vec.append(hid_ent1_g * batch_ent1_g_mask + (1 - batch_ent1_g_mask) * self.rand_emb(self.the_one))
        # print("1")
        # get_gpu_mem_info()

        if "m" in self.model_type:
            cm_out = self.cm_attn(batch_ent1_g, hid_ent1_d, batch_ent1_d_mask, batch_ent1_g_mask)
            final_vec.append(cm_out)
        # print("2")
        #
        # get_gpu_mem_info()
        "=========Classification=========="

        # final_vec = torch.cat(final_vec, dim=-1)
        # final_vec = torch.cat([cm_out, ent1_embeds, ent1_mask_embeds], dim=-1)
        # print("final_vec",get_tensor_info(final_vec))

        output = self.combiner(torch.cat(final_vec, dim=-1))

        # print("output",get_tensor_info(output))

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
