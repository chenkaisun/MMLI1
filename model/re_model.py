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


class RE(torch.nn.Module):
    def __init__(self, args):
        super(RE, self).__init__()

        self.model_type = args.model_type

        if 'g' in args.model_type:
            # self.gnn = MoleGNN(args)
            self.gnn = MoleGNN2(args)

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
            self.combiner = Linear(args.plm_hidden_dim * 5 , args.out_dim)
        elif 'tdg' in args.model_type:
            print("args.tdg")
            # self.combiner = Linear(args.plm_hidden_dim * 3 + final_dim, args.out_dim)
            self.combiner = Linear(args.plm_hidden_dim * 5 , args.out_dim)
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

        # self.map2smaller = Linear(args.plm_hidden_dim, args.g_dim)
        self.text_transform = Linear(args.plm_hidden_dim, args.g_dim)

        self.criterion = LabelSmoothingCrossEntropy(reduction='sum')
        # self.dropout = torch.nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.loss = torch.nn.CrossEntropyLoss()
        self.cm_attn = CrossModalAttention(reduction='mean', m1_dim=args.g_dim, m2_dim=args.plm_hidden_dim,
                                           final_dim=final_dim)

    def forward(self, input, args):
        # texts = input['texts']
        # texts_mask = input['texts_mask']
        #
        # batch_ent1_d = input['batch_ent1_d']
        # batch_ent1_d_mask = input['batch_ent1_d_mask']
        # batch_ent2_d = input['batch_ent2_d']
        # batch_ent2_d_mask = input['batch_ent2_d_mask']
        # batch_ent1_g = input['batch_ent1_g']
        # batch_ent1_g_mask = input['batch_ent1_g_mask']
        # ids = input['ids']
        # labels = input['labels']
        # in_train = input['in_train']
        texts = input.texts
        texts_attn_mask = input.texts_attn_mask
        labels = input.labels
        # xi[xi != xi] = 0
        batch_ent1_d = input.ent1_d
        # print("batch_ent1_d",batch_ent1_d)
        batch_ent1_d_mask = input.ent1_d_mask
        batch_ent2_d = input.ent2_d
        # print("batch_ent2_d",batch_ent2_d)
        batch_ent2_d_mask = input.ent2_d_mask
        batch_ent1_g = input.ent1_g
        batch_ent1_g_mask = input.ent1_g_mask

        ent1_pos = input.ent1_pos
        ent2_pos = input.ent2_pos
        concepts = input.concepts


        in_train = input.in_train

        # print("batch_graph_data", batch_graph_data)

        # bert
        # token_id_offset = 1
        # modals = []
        hid_texts, hid_ent1_d, hid_ent2_d, hid_ent1_g, hid_ent2_g = None, None, None, None, None
        final_vec = []

        if 'tdgx' not in self.model_type:
            # print('not tdgx')
            # if 't' in args.model_type:
            # hid_texts = self.plm(**texts, return_dict=True).pooler_output
            # hid_texts = self.plm(**texts, return_dict=True).last_hidden_state[:, 0, :]
            # print("texts", texts)
            hid_texts = self.plm(input_ids=texts, attention_mask=texts_attn_mask,
                                 return_dict=True).last_hidden_state[:, :, :]
            # hid_texts=self.text_transform(hid_texts)
            # hid_texts=F.gelu(hid_texts)
            # hid_texts = self.plm(input_ids=texts, attention_mask=texts_attn_mask,
            #                      return_dict=True).last_hidden_state[:, :, :]
            # # print("hid_texts", get_tensor_info(hid_texts))
            # print("hid_texts",hid_texts)
            #
            # hid_texts = F.dropout(hid_texts, self.dropout, training=self.training)
            # print("dropout hid_texts",hid_texts)
            #
            ent1_embeds, ent2_embeds = [], []
            for i in range(hid_texts.shape[0]):
                text_embed = hid_texts[i]
                # print("text_embed", get_tensor_info(text_embed))

                ent1_embeds.append(text_embed[ent1_pos[i, 0]])
                ent2_embeds.append(text_embed[ent2_pos[i, 0]])
                # print("text_embed[ent1_pos[i, 0]]", get_tensor_info(text_embed[ent1_pos[i, 0]]))
            ent1_embeds = torch.stack(ent1_embeds, dim=0)  # [n_e, d]
            # print("ent1_embeds", get_tensor_info(ent1_embeds))
            ent2_embeds = torch.stack(ent2_embeds, dim=0)  # [n_e, d]

            concepts_emb=self.plm(**concepts, return_dict=True).last_hidden_state[:, 0, :]

            ent1_embeds+=concepts_emb[0]
            ent2_embeds+=concepts_emb[1]
            # print("ent1_embeds",ent1_embeds)
            # print("ent2_embeds",ent2_embeds)concepts

            #
            # print("ent1_embeds", get_tensor_info(ent1_embeds))
            # print("ent2_embeds", get_tensor_info(ent2_embeds))
            # print()
            # modals.append(hid_texts)
            if 'd' in self.model_type:
                # print("in d")
                # try lstm
                # 1:-1
                hid_ent1_d = self.plm(**batch_ent1_d, return_dict=True).last_hidden_state[:, :, :]  # *batch_ent1_d_mask
                # hid_ent1_d = self.plm(**batch_ent1_d, return_dict=True).pooler_output  # *batch_ent1_d_mask


                # hid_ent1_d = self.plm(**batch_ent1_d, return_dict=True).last_hidden_state[:, 0, :]  # *batch_ent1_d_mask

                # hid_ent1_d= self.text_transform(hid_ent1_d)
                # # # # if args.mult_mask: hid_ent1_d *= batch_ent1_d_mask
                # hid_ent1_d=torch.tanh(hid_ent1_d)

                # hid_ent1_d = F.dropout(hid_ent1_d, self.dropout, training=self.training)

                # print("hid_ent1_d", get_tensor_info(hid_ent1_d))

                # print("d hid_ent1_d", hid_ent1_d)
                # hid_ent2_d = self.plm(**batch_ent2_d, return_dict=True).last_hidden_state[:, 0, :]  # *batch_ent2_d_mask

                # hid_ent1_d= self.text_transform(hid_ent2_d)
                # # # # if args.mult_mask: hid_ent1_d *= batch_ent1_d_mask
                # hid_ent2_d=torch.tanh(hid_ent2_d)

                # if args.mult_mask: hid_ent2_d *= batch_ent2_d_mask
                # hid_ent2_d=F.gelu(hid_ent2_d)
                # hid_ent2_d = F.dropout(hid_ent2_d, self.dropout, training=self.training)

                # print("hid_ent2_d", get_tensor_info(hid_ent2_d))

                # print("d hid_ent2_d", hid_ent2_d)

                if not 'g' in self.model_type:
                    # ent1_embeds = torch.cat([ent1_embeds, hid_ent1_d], dim=-1)
                    ent1_embeds = torch.cat([ent1_embeds, hid_ent1_d[:, 0, :]], dim=-1)

                # ent2_embeds = torch.cat([ent2_embeds, hid_ent2_d], dim=-1)
                # print("ent2_embeds 2", get_tensor_info(ent2_embeds))

                # modals.extend([hid_ent1_d, hid_ent2_d])
            if 'g' in self.model_type:
                # print("batch_ent1_g", batch_ent1_g)

                # B'xd
                hid_ent1_g = self.gnn(batch_ent1_g, args.g_global_pooling)  # * batch_ent1_g_mask
                # print("hid_ent1_g", hid_ent1_g)

                # if args.g_global_pooling:
                #     if args.g_mult_mask: hid_ent1_g *= batch_ent1_g_mask
                # hid_ent2_g = self.gnn(batch_ent2_g)# * batch_ent2_g_mask
                # print("gnn hid_ent1_g", hid_ent1_g)
                # print("gnn hid_ent2_g", hid_ent2_g)
                # modals.extend([hid_ent1_g])

                if 'd' in self.model_type:
                    d_modal, g_modal = hid_ent1_d, hid_ent1_g
                    # print("hid_ent1_d", hid_ent1_d)
                    # print("hid_ent1_g", hid_ent1_g)

                    tmp_batch = []
                    # graph_ids = torch.unique(batch_ent1_g.batch).cpu().numpy().astype(int)
                    # print("graph_ids",graph_ids)graph_ids
                    # batch_ids = batch_ent1_g.batch.cpu().numpy().astype(int)
                    for graph_id in range(hid_ent1_d.shape[0]):
                        # print("graph_id", graph_id)
                        # print("g_modal[batch_ent1_g.batch == graph_id]",
                        #       get_tensor_info(g_modal[batch_ent1_g.batch == graph_id]))
                        # print("d_modal[graph_id]", get_tensor_info(d_modal[graph_id]))
                        # print("g_modal[batch_ent1_g.batch == graph_id]", get_tensor_info(g_modal[batch_ent1_g.batch == graph_id]))
                        # print("d_modal", get_tensor_info(d_modal))
                        # print("d_modal[graph_id]", get_tensor_info(d_modal[graph_id]))

                        # g_indices=[i for i, bl in enumerate(batch_ent1_g.batch == graph_id) if bl]
                        # print("g_indices", g_indices)
                        # out = self.cm_attn(torch.index_select(g_modal, 0,
                        #                    torch.tensor(g_indices, dtype=torch.long, device=g_modal.device)),
                        #                    torch.index_select(d_modal, 0,
                        #                                       torch.tensor([graph_id], dtype=torch.long,
                        #                                                    device=g_modal.device)).squeeze())

                        # tmp rmv
                        # out = self.cm_attn(g_modal[batch_ent1_g.batch == graph_id], d_modal[graph_id])

                        # tmp_batch.append(out)
                        # tmp_batch.append(torch.cat([hid_ent1_d[graph_id, 0, :]], dim=-1))
                        # tmp_batch.append(torch.cat([out, hid_ent1_d[graph_id, 0, :]], dim=-1))

                        # NO CM
                        g_feat= (g_modal[graph_id] * batch_ent1_g_mask[graph_id, 0]) if args.g_mult_mask else g_modal[graph_id]
                        d_feat=d_modal[graph_id, 0, :] * batch_ent1_d_mask[graph_id, 0] if args.mult_mask else d_modal[graph_id, 0, :]


                        tmp_batch.append(torch.cat([g_feat,d_feat], dim=-1))

                        # tmp_batch.append(torch.cat([g_modal[graph_id] * batch_ent1_g_mask[graph_id, 0],
                        #                             d_modal[graph_id, 0, :] * batch_ent1_d_mask[graph_id, 0]], dim=-1))

                        # print("batch_ent1_g_mask[graph_id, 0]", batch_ent1_g_mask[graph_id, 0])
                        #
                        # print("tmp_batch[-1].shape", get_tensor_info(tmp_batch[-1]))
                        # indices = np.argwhere(batch_ids == graph_id).flatten()
                        # print("indices", indices)
                        # max_num_nodes = max(max_num_nodes, len(indices))
                        # graph_list.append(torch.index_select(x, 0, torch.LongTensor(indices).cuda()))
                    # print("torch.stack(tmp_batch, dim=0)", get_tensor_info(torch.stack(tmp_batch, dim=0)))

                    ent1_embeds = torch.cat([ent1_embeds, torch.stack(tmp_batch, dim=0)], dim=-1)
                else:
                    ent1_embeds = torch.cat([ent1_embeds, hid_ent1_g], dim=-1)

            # print("ent1_embeds", get_tensor_info(ent1_embeds))
            # print("ent2_embeds", get_tensor_info(ent2_embeds))

            # final_vec = torch.cat([hid_texts[:,0,:], ent1_embeds, ent2_embeds], dim=-1)
            final_vec = torch.cat([hid_texts[:, 0, :], ent1_embeds, ent2_embeds], dim=-1)

            # final_vec = hid_texts#[:,0,:]

            # hid_texts
            # print("final_vec", final_vec)
            # modals[1] = final_vec
            # modals.pop()
        # else:
        #     hid_texts = self.plm(**texts, return_dict=True).last_hidden_state
        #     hid_texts = self.dropout(hid_texts)
        #     # modals.append(hid_texts)
        #     hid_ent1_d = self.plm(**batch_ent1_d, return_dict=True).last_hidden_state
        #     hid_ent1_d = self.dropout(hid_ent1_d)
        #     hid_ent2_d = self.plm(**batch_ent2_d, return_dict=True).last_hidden_state
        #     hid_ent2_d = self.dropout(hid_ent2_d)
        #     hid_ent1_gs = self.gnn(batch_ent1_g, global_pooling=False)
        #     # hid_ent2_gs = self.gnn(batch_ent2_g, global_pooling=False)
        #     # print('e0')
        #     # print(hid_ent1_gs.shape)
        #     # print(hid_ent2_gs.shape)
        #     res_g1 = []
        #     res_g2 = []
        #     res_d1 = []
        #     res_d2 = []
        #     for i, (hid_ent1_g, hid_ent2_g) in enumerate(zip(hid_ent1_gs, hid_ent2_gs)):
        #         print("i", i)
        #         cur_hid_ent1_d = hid_ent1_d[i, :, :]
        #         cur_hid_ent2_d = hid_ent2_d[i, :, :]
        #         tmp_hid_ent1_d, tmp_hid_ent2_d = self.map2smaller(cur_hid_ent1_d), self.map2smaller(cur_hid_ent2_d)
        #         x_attn_mat1, x_attn_mat2 = hid_ent1_g.mm(tmp_hid_ent1_d.t()), hid_ent2_g.mm(tmp_hid_ent2_d.t())
        #         cur_hid_ent1_g, cur_hid_ent2_g = x_attn_mat1.mm(cur_hid_ent1_d), x_attn_mat2.mm(cur_hid_ent2_d)
        #         cur_hid_ent1_d, cur_hid_ent2_d = x_attn_mat1.t().mm(hid_ent1_g), x_attn_mat2.t().mm(hid_ent2_g)
        #         res_g1.append(cur_hid_ent1_g)
        #         res_g2.append(cur_hid_ent2_g)
        #         res_d1.append(cur_hid_ent1_d)
        #         res_d2.append(cur_hid_ent2_d)
        #         print(hid_texts.shape)
        #         print(cur_hid_ent1_g.shape)
        #         print(cur_hid_ent2_g.shape)
        #         print(cur_hid_ent1_d.shape)
        #         print(cur_hid_ent2_d.shape)
        #     modals.extend([hid_texts, torch.cat(res_g1, dim=0), torch.cat(res_g2, dim=0),
        #                    torch.cat(res_d1, dim=0), torch.cat(res_d1, dim=0)])
        #
        #     #
        #     # tmp_hid_ent1_d, tmp_hid_ent2_d = self.map2smaller(hid_ent1_d), self.map2smaller(hid_ent2_d)
        #     # x_attn_mat1, x_attn_mat2 = hid_ent1_g.mm(tmp_hid_ent1_d.t()), hid_ent2_g.mm(tmp_hid_ent2_d.t())
        #     # hid_ent1_g, hid_ent2_g = x_attn_mat1.mm(hid_ent1_d), x_attn_mat2.mm(hid_ent2_d)
        #     # hid_ent1_d, hid_ent2_d = x_attn_mat1.t().mm(hid_ent1_g), x_attn_mat2.t().mm(hid_ent2_g)
        #     # modals.extend([hid_texts, hid_ent1_g, hid_ent2_g, hid_ent1_d, hid_ent2_d])
        #
        #     # relu ?

        # output = self.combiner(F.dropout(final_vec, self.dropout, training=self.training))

        output = self.combiner(final_vec)

        #
        # if args.t:
        #     # hid_texts = self.plm(**texts, return_dict=True).pooler_output
        #     hid_texts = self.plm(**texts, return_dict=True).last_hidden_state[:, 0, :]
        #     pooled = self.dropout(hid_texts)
        #     output = self.combiner(pooled)
        #     # output = torch.softmax(output, dim=-1)
        # elif args.td:
        #     # hid_texts = self.plm(**texts, return_dict=True).pooler_output
        #
        #     hid_texts = self.plm(**texts, return_dict=True).last_hidden_state[:, 0, :]
        #     pooled = self.dropout(hid_texts)
        #     output = self.combiner(pooled)
        #
        #
        #     hid_texts = self.plm(**texts, return_dict=True).last_hidden_state[:, 0, :]
        #     hid_texts = self.dropout(hid_texts)
        #     hid_ent1_d = self.plm(**batch_ent1_d, return_dict=True).pooler_output * batch_ent1_d_mask
        #     # print("hid_ent1_d", get_tensor_info(hid_ent1_d))
        #     hid_ent2_d = self.plm(**batch_ent2_d, return_dict=True).pooler_output * batch_ent2_d_mask
        #     hid_ent1_g = self.gnn(batch_ent1_g) * batch_ent1_g_mask
        #     hid_ent2_g = self.gnn(batch_ent2_g) * batch_ent2_g_mask

        # print("output",output.shape)
        # print("batch_graph_data.y.unsqueeze(-1)",batch_graph_data.y.shape)

        # print("self.training", self.training)
        if in_train:
            # label smoothing
            # return self.criterion(output, labels)
            return self.loss(output, labels)
            return torch.nn.functional.cross_entropy(output, labels)
            return torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)

        pred_out = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
        # print('pred_out', get_tensor_info(pred_out))
        return pred_out

        return F.log_softmax(x, dim=1)
