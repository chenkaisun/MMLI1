import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# from torch import optim


def get_tensor_info(tensor):
    return f"Shape: {tensor.shape} | Type: {tensor.type()} | Device: {tensor.device}"


def get_loss_fn(loss_name):
    if loss_name == "mse":
        return torch.nn.MSELoss()
    elif loss_name == "bce":
        return torch.nn.BCELoss()
    elif loss_name == "bce_logit":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_name == "ce":
        return torch.nn.CrossEntropyLoss()
    elif loss_name == "kl":
        return torch.nn.KLDivLoss()
    elif loss_name == "nll":
        return torch.nn.NLLLoss()
    else:
        assert False, f"loss_name {loss_name} not valid"


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class CrossModalAttention(nn.Module):
    def __init__(self, reduction='mean', m1_dim=0, m2_dim=0, final_dim=800):
        super().__init__()
        # self.epsilon = epsilon
        self.reduction = reduction
        self.temprature = 3
        # self.aggregate = True
        self.l_filter1 = torch.nn.Linear(m1_dim, final_dim)
        self.l_filter2 = torch.nn.Linear(m2_dim, final_dim)

        self.l1 = torch.nn.Linear(m1_dim, final_dim)
        self.l2 = torch.nn.Linear(m2_dim, final_dim)

    def forward(self, m1, m2, aggregate=True):
        """
        :param m1: vectors of a modality
        :param m2: same as above
        :return: attended embeddings
        """
        # print("m1", get_tensor_info(m1))
        # print("m2", get_tensor_info(m2))
        # print("m1", m1)
        # print("m2", m2)

        # norms1, norms2 = torch.norm(m1, dim=-1, p=2, keepdim=True), torch.norm(m2, dim=-1, p=2, keepdim=True)
        # # print("norms1, norms2", norms1, norms2)
        #
        # norm_prod_mat = norms1.matmul(norms2.t())
        # # print("norm_prod_mat.shape", norm_prod_mat.shape)
        raw_prod = m1.matmul(m2.t())
        # # print("raw_prod", get_tensor_info(raw_prod))
        # # print("raw_prod", raw_prod)
        #
        # cos_sim_mat = raw_prod / norm_prod_mat
        # # print("cos_sim_mat", cos_sim_mat)
        # c_plus = torch.relu(cos_sim_mat)
        # # print("c_plus", c_plus)
        #
        # # c_hat=torch.tensor()
        # # sum_each_row = c_plus.pow(2).sum(dim=-1, keepdim=True).sqrt()
        # # sum_each_row=torch.sqrt(torch.sum(torch.pow(c_hat, 2), dim=-1, keepdim=True))
        # c_hat = c_plus / torch.norm(c_plus, dim=-1, p=2, keepdim=True)  # / sum_each_row
        # print("c_hat", c_hat)

        # print("c_hat", get_tensor_info(c_hat))
        # + torch.tensor(1e-7, device=c_hat.device)
        # alpha = torch.softmax(self.temprature * (c_hat.t()+torch.tensor(1e-7, device=c_hat.device)), dim=-1)  # .t()
        # print("self.temprature*(c_hat.t()+torch.tensor(1e-7, device=c_hat.device)",
        #       self.temprature * (c_hat.t() + torch.tensor(1e-7, device=c_hat.device)))
        alpha2 = torch.softmax(raw_prod, dim=-1)
        alpha1 = torch.softmax(raw_prod.t(), dim=-1)  # + torch.tensor(1e-9, device=m1.device)
        # print("alpha", alpha)

        attended_m1 = alpha1.matmul(m1)
        attended_m2 = alpha2.matmul(m2)
        # print("attended_m1.mean(0)", get_tensor_info(attended_m1.mean(0)))

        # return attended_m1.mean(0)
        # print("attended_m1", get_tensor_info(attended_m1))
        # print("attended_m1", attended_m1)
        # return F.tanh(self.l1(attended_m1.sum(0)))
        if aggregate:
            attended_m1 = attended_m1
            attended_m2 = attended_m2

            # m2 = m2.mean(0)
            # print("numnan", torch.sum(torch.isnan(torch.cat([attended_m1, m2], dim=-1))))

            attended_m1 = F.dropout(attended_m1, 0.1, training=self.training)
            # m2 = F.dropout(m2, 0.1, training=self.training)
            attended_m2 = F.dropout(attended_m2, 0.1, training=self.training)
            # filter = torch.sigmoid(self.l_filter(torch.cat([attended_m1, m2], dim=-1)))  # .sum(0)
            filter1 = torch.sigmoid(self.l_filter1(torch.cat([attended_m1], dim=-1)).mean(0))  # .sum(0)
            filter2 = torch.sigmoid(self.l_filter2(torch.cat([attended_m2], dim=-1)).mean(0))  # .sum(0)

            # print("dropout attended_m1", attended_m1)
            # print("dropout m2", m2)
            # print("filter", filter)
            # return attended_m1.sum(0)

            transformed_m1 = torch.tanh(self.l1(attended_m1))
            transformed_m2 = torch.tanh(self.l2(attended_m2))
            # print("transformed_m1", transformed_m1)
            # print("transformed_m2", transformed_m2)
            # print("filter", get_tensor_info(filter))
            # print("transformed_m1", get_tensor_info(transformed_m1))
            # print("transformed_m2", get_tensor_info(transformed_m2))

            return transformed_m1 * filter1 + transformed_m2 * filter2

        return attended_m1, m2


def encode(self, input_ids, attention_mask):
    config = self.config
    if config.transformer_type == "bert":
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id]
    elif config.transformer_type == "roberta":
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id, config.sep_token_id]
    sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
    return sequence_output, attention


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention
