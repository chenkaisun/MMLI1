import torch
import torch.nn.functional as F
import torch.utils.data
from torch.nn import Linear
from transformers import AutoModel

from model.gnn import MoleGNN


class JNet(torch.nn.Module):
    def __init__(self, args):
        super(JNet, self).__init__()
        self.gnn = MoleGNN(args)
        self.plm = AutoModel.from_pretrained(args.plm)
        if args.g_only:
            self.combiner = Linear(args.g_dim, 1)
        elif args.t_only:
            self.combiner = Linear(args.plm_hidden_dim, 1)
        else:
        # self.combiner = Linear(args.g_dim, 1)
        # self.combiner = Linear(args.plm_hidden_dim, 1)
            self.combiner = Linear(args.plm_hidden_dim + args.g_dim, 1)
        self.dropout = args.dropout

    def forward(self, input, args):
        input_ids = input['input_ids']
        batch_graph_data = input['batch_graph_data']
        ids = input['ids']
        in_train = input['in_train']
        # print("batch_graph_data", batch_graph_data)

        if args.g_only:
            hid = self.gnn(batch_graph_data)
        elif args.t_only:
            hid = self.plm(**input_ids, return_dict=True).pooler_output
        else:
            hid = torch.cat([self.gnn(batch_graph_data), self.plm(**input_ids, return_dict=True).pooler_output], dim=-1)
        # print("g_out", g_out.shape)
        # print("batch_graph_data", batch_graph_data.shape)

        # t_out=self.plm(**input_ids, return_dict=True).pooler_output
        # print(get_tensor_info(g_out))
        # print(get_tensor_info(t_out))
        # print("g_out", get_tensor_info(g_out))
        # print("t_out", get_tensor_info(t_out))

        # output=self.combiner(torch.cat([g_out, t_out], dim=-1))
        # output=self.combiner(torch.cat([t_out], dim=-1))
        output = self.combiner(hid)

        # print("output",output.shape)
        # print("batch_graph_data.y.unsqueeze(-1)",batch_graph_data.y.shape)
        if in_train:
            return torch.nn.functional.binary_cross_entropy_with_logits(output, batch_graph_data.y.view(-1, 1))
        # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)
        return torch.sigmoid(output)

        return F.log_softmax(x, dim=1)
