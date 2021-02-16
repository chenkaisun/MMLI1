import torch
import torch.nn
from torch.nn import functional as F
from torch_geometric.nn import BatchNorm, GATConv, global_mean_pool


class MoleGNN(torch.nn.Module):
    def __init__(self, args):
        super(MoleGNN, self).__init__()
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
        print("args.in_dim", args.in_dim)
        self.conv1 = GATConv(args.in_dim, 32, dropout=self.dropout)
        self.conv2 = GATConv(32, 32, dropout=self.dropout)
        self.conv3 = GATConv(32, 32, dropout=self.dropout)
        self.conv4 = GATConv(32, 32, dropout=self.dropout)
        self.conv5 = GATConv(32, args.g_dim, dropout=self.dropout)
        # self.conv1 = GATConv(5, 32)
        # self.conv2 = GATConv(32, 32)
        # self.conv3 = GATConv(32, 32)
        # self.conv4 = GATConv(32, 32)
        # self.conv5 = GATConv(32, 32)
        # self.conv3 = GATConv(128, 256)
        # self.lin = Linear(16, 2)

    def forward(self, data):
        # print("data", data.x.shape)
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
        torch.nn.functional.tanh(x)
        # x=self.mid(x)
        x=self.conv2(x, edge_index)
        torch.nn.functional.tanh(x)
        # x=self.mid(x)
        # x=self.mid(x)
        x=self.conv3(x, edge_index)
        x=self.conv4(x, edge_index)
        # x=self.conv5(x, edge_index)

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