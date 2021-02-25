import torch
import torch.nn.functional as F
import torch.utils.data
from torch.nn import Linear
from transformers import AutoModel

from model.gnn import MoleGNN
# from train_utils import get_tensor_info
from model.model_utils import get_tensor_info, LabelSmoothingCrossEntropy
class RE(torch.nn.Module):
    def __init__(self, args):
        super(RE, self).__init__()

        if not args.bert_only:
            self.gnn = MoleGNN(args)
        self.plm = AutoModel.from_pretrained(args.plm)
        # if args.g_only:
        #     self.combiner = Linear(args.g_dim, 1)
        # elif args.t_only:
        #     self.combiner = Linear(args.plm_hidden_dim, 1)
        # else:
        # self.combiner = Linear(args.g_dim, 1)
        # self.combiner = Linear(args.plm_hidden_dim, 1)
        if args.bert_only:
            self.combiner = Linear(args.plm_hidden_dim, args.out_dim)
        else: self.combiner = Linear(args.plm_hidden_dim * 3 + 2 * args.g_dim, args.out_dim)
        self.criterion = LabelSmoothingCrossEntropy(reduction='sum')
        self.dropout = args.dropout

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

        if args.bert_only:
            hid_texts = self.plm(**texts, return_dict=True).pooler_output
            output = self.combiner(torch.cat([hid_texts], dim=-1))

        else:
            hid_texts = self.plm(**texts, return_dict=True).pooler_output
            hid_ent1_d = self.plm(**batch_ent1_d, return_dict=True).pooler_output * batch_ent1_d_mask
            # print("hid_ent1_d", get_tensor_info(hid_ent1_d))
            hid_ent2_d = self.plm(**batch_ent2_d, return_dict=True).pooler_output * batch_ent2_d_mask
            hid_ent1_g = self.gnn(batch_ent1_g) * batch_ent1_g_mask
            hid_ent2_g = self.gnn(batch_ent2_g) * batch_ent2_g_mask


            # relu ?
            output = self.combiner(torch.cat([hid_texts, hid_ent1_d, hid_ent1_g, hid_ent2_d, hid_ent2_g], dim=-1))

        # print("output",output.shape)
        # print("batch_graph_data.y.unsqueeze(-1)",batch_graph_data.y.shape)

        # print("self.training", self.training)
        if in_train:

            # label smoothing
            return self.criterion(output, labels)

            # return torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)

        pred_out=torch.argmax(torch.softmax(output, dim=-1), dim=-1)
        # print('pred_out', get_tensor_info(pred_out))
        return pred_out

        return F.log_softmax(x, dim=1)
