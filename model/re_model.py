import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
from torch.nn import Linear
from transformers import AutoModel

from model.gnn import MoleGNN
# from train_utils import get_tensor_info
from model.model_utils import get_tensor_info, LabelSmoothingCrossEntropy


class RE(torch.nn.Module):
    def __init__(self, args):
        super(RE, self).__init__()

        if 'g' in args.model_type:
            self.gnn = MoleGNN(args)

        self.plm = AutoModel.from_pretrained(args.plm)
        # if args.g_only:
        #     self.combiner = Linear(args.g_dim, 1)
        # elif args.t_only:
        #     self.combiner = Linear(args.plm_hidden_dim, 1)
        # else:
        # self.combiner = Linear(args.g_dim, 1)
        # self.combiner = Linear(args.plm_hidden_dim, 1)
        if 't' in args.model_type:
            print("args.t")
            self.combiner = Linear(args.plm_hidden_dim, args.out_dim)
        if 'td' in args.model_type:
            print("args.td")
            self.combiner = Linear(args.plm_hidden_dim * 3, args.out_dim)
        if 'tg' in args.model_type:
            print("args.tg")
            self.combiner = Linear(args.plm_hidden_dim + 2 * args.g_dim, args.out_dim)
        if 'tdg' in args.model_type:
            print("args.tdg")
            self.combiner = Linear(args.plm_hidden_dim * 3 + 2 * args.g_dim, args.out_dim)
        if 'tdgx' in args.model_type:
            print("tdgx")
            self.combiner = Linear(args.plm_hidden_dim * 3 + 2 * args.g_dim, args.out_dim)

        self.map2smaller = Linear(args.plm_hidden_dim, args.g_dim)
        self.criterion = LabelSmoothingCrossEntropy(reduction='sum')
        self.dropout = torch.nn.Dropout(args.dropout)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input, args):
        texts = input['texts']
        batch_ent1_d = input['batch_ent1_d']
        batch_ent1_d_mask = input['batch_ent1_d_mask']
        batch_ent2_d = input['batch_ent2_d']
        batch_ent2_d_mask = input['batch_ent2_d_mask']
        batch_ent1_g = input['batch_ent1_g']
        batch_ent1_g_mask = input['batch_ent1_g_mask']
        batch_ent2_g = input['batch_ent2_g']
        batch_ent2_g_mask = input['batch_ent2_g_mask']
        ids = input['ids']
        labels = input['labels']
        in_train = input['in_train']
        # print("batch_graph_data", batch_graph_data)

        modals=[]
        hid_texts, hid_ent1_d, hid_ent2_d, hid_ent1_g, hid_ent2_g = None, None, None, None, None
        if 'tdgx' not in args.model_type:
            print('not tdgx')
            if 't' in args.model_type:
                # hid_texts = self.plm(**texts, return_dict=True).pooler_output
                hid_texts = self.plm(**texts, return_dict=True).last_hidden_state[:, 0, :]
                hid_texts = self.dropout(hid_texts)
                modals.append(hid_texts)
            if 'd' in args.model_type:
                hid_ent1_d = self.plm(**batch_ent1_d, return_dict=True).last_hidden_state[:, 0, :]
                hid_ent1_d = self.dropout(hid_ent1_d)
                hid_ent2_d = self.plm(**batch_ent2_d, return_dict=True).last_hidden_state[:, 0, :]
                hid_ent2_d = self.dropout(hid_ent2_d)
                modals.extend([hid_ent1_d, hid_ent2_d])
            if 'g' in args.model_type:
                print("batch_ent1_g", batch_ent1_g)
                hid_ent1_g = self.gnn(batch_ent1_g) * batch_ent1_g_mask
                hid_ent2_g = self.gnn(batch_ent2_g) * batch_ent2_g_mask
                print("gnn hid_ent1_g", hid_ent1_g)
                print("gnn hid_ent2_g", hid_ent2_g)

                modals.extend([hid_ent1_g, hid_ent2_g])
        else:
            hid_texts = self.plm(**texts, return_dict=True).last_hidden_state
            hid_texts = self.dropout(hid_texts)
            # modals.append(hid_texts)
            hid_ent1_d = self.plm(**batch_ent1_d, return_dict=True).last_hidden_state
            hid_ent1_d = self.dropout(hid_ent1_d)
            hid_ent2_d = self.plm(**batch_ent2_d, return_dict=True).last_hidden_state
            hid_ent2_d = self.dropout(hid_ent2_d)
            hid_ent1_gs = self.gnn(batch_ent1_g, global_pooling=False)
            hid_ent2_gs = self.gnn(batch_ent2_g, global_pooling=False)
            print('e0')
            print(hid_ent1_gs.shape)
            print(hid_ent2_gs.shape)
            res_g1=[]
            res_g2=[]
            res_d1=[]
            res_d2=[]
            for i, (hid_ent1_g, hid_ent2_g) in enumerate(zip(hid_ent1_gs, hid_ent2_gs)):
                print("i", i)
                cur_hid_ent1_d = hid_ent1_d[i, :, :]
                cur_hid_ent2_d = hid_ent2_d[i, :, :]
                tmp_hid_ent1_d, tmp_hid_ent2_d = self.map2smaller(cur_hid_ent1_d), self.map2smaller(cur_hid_ent2_d)
                x_attn_mat1, x_attn_mat2 = hid_ent1_g.mm(tmp_hid_ent1_d.t()), hid_ent2_g.mm(tmp_hid_ent2_d.t())
                cur_hid_ent1_g, cur_hid_ent2_g = x_attn_mat1.mm(cur_hid_ent1_d), x_attn_mat2.mm(cur_hid_ent2_d)
                cur_hid_ent1_d, cur_hid_ent2_d = x_attn_mat1.t().mm(hid_ent1_g), x_attn_mat2.t().mm(hid_ent2_g)
                res_g1.append(cur_hid_ent1_g)
                res_g2.append(cur_hid_ent2_g)
                res_d1.append(cur_hid_ent1_d)
                res_d2.append(cur_hid_ent2_d)
                print(hid_texts.shape)
                print(cur_hid_ent1_g.shape)
                print(cur_hid_ent2_g.shape)
                print(cur_hid_ent1_d.shape)
                print(cur_hid_ent2_d.shape)
            modals.extend([hid_texts, torch.cat(res_g1, dim=0), torch.cat(res_g2, dim=0),
                           torch.cat(res_d1, dim=0), torch.cat(res_d1, dim=0)])

            #
            # tmp_hid_ent1_d, tmp_hid_ent2_d = self.map2smaller(hid_ent1_d), self.map2smaller(hid_ent2_d)
            # x_attn_mat1, x_attn_mat2 = hid_ent1_g.mm(tmp_hid_ent1_d.t()), hid_ent2_g.mm(tmp_hid_ent2_d.t())
            # hid_ent1_g, hid_ent2_g = x_attn_mat1.mm(hid_ent1_d), x_attn_mat2.mm(hid_ent2_d)
            # hid_ent1_d, hid_ent2_d = x_attn_mat1.t().mm(hid_ent1_g), x_attn_mat2.t().mm(hid_ent2_g)
            # modals.extend([hid_texts, hid_ent1_g, hid_ent2_g, hid_ent1_d, hid_ent2_d])


            # relu ?
        output = self.combiner(torch.cat(modals, dim=-1))


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
