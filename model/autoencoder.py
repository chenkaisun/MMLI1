import torch
import torch.nn.functional as F
import torch.utils.data
from torch.nn import Linear
from transformers import AutoModel

from model.gnn import MoleGNN


class AE(torch.nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.gnn = MoleGNN(args)
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
