from torch.utils.data import DataLoader
import torch
from torch import nn
from data import collate_fn
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error
import numpy as np
def evaluate(args, model, data):
    # print("Evaluate")

    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    targets = []
    for batch in dataloader:
        model.eval()

        encoded_input = batch[0]  # tokenizer(batch[0])
        inputs = {'input_ids': {key: encoded_input[key].to(args.device) for key in encoded_input},
                  'batch_graph_data': batch[1].to(args.device),
                  'ids': batch[2],
                  'in_train': False,
                  }
        with torch.no_grad():
            pred = model(inputs, args)
            # print(pred.shape)
            # print(pred.cpu())
            # print(pred.cpu().numpy())
            # print(pred.cpu().numpy().squeeze(-1))
            pred = list(pred.cpu().numpy().squeeze(-1))
            preds.extend(pred)

            # print(inputs["ids"], inputs["batch_graph_data"].y, inputs["batch_graph_data"].y.cpu().numpy().squeeze(-1), list(inputs["batch_graph_data"].y.cpu().numpy().squeeze(-1)))
            # print("ergf", list(inputs["batch_graph_data"].y.cpu().numpy().squeeze()))
            targets.extend(list(inputs["batch_graph_data"].y.cpu().numpy()))
            # print("new targets", targets)

    # print("preds  ", preds)
    # print("targets", targets)
    # score = roc_auc_score(targets, preds)

    preds=np.array(preds)
    # print(targets, preds.tolist())
    score2 = roc_auc_score(targets, preds.tolist())
    preds[preds>=0.5]=1
    preds[preds<0.5]=0
    score = accuracy_score(targets, preds.tolist())
    args.logger.debug(f"accuracy_score {score}")
    args.logger.debug(f"auc_score {score2}", )
    # f1 = f1_score(targets, preds, average='macro')
    output = None
    return score2, output
