from torch.utils.data import DataLoader
# import torch
# from torch import nn
from data import collate_fn
import torch
# import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error, \
    precision_score, recall_score
import numpy as np
from data import collate_wrapper, collate_fn
from pprint import pprint as pp
from sklearn.metrics import accuracy_score

a = np.array([[1, 1], [0, 1]])
b = np.array([[1, 1], [1, 0]])

accuracy_score(a, b)


def get_prf(targets, preds, average="micro", verbose=False):
    precision = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(targets, preds, average=average)
    # print(precision, recall, f1)
    if verbose: print(f"{average}: precision {precision} recall {recall} f1 {f1}")
    return precision, recall, f1



def evaluate(args, model, data):
    # print("Evaluate")

    if args.exp == "mol_pred":

        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                drop_last=False)
    else:

        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper,
                                drop_last=False)
    preds = []
    targets = []
    ids = []
    for batch in dataloader:
        model.eval()

        if args.exp == "mol_pred":
            encoded_input = batch[0]  # tokenizer(batch[0])
            inputs = {'input_ids': {key: encoded_input[key].to(args.device) for key in encoded_input},
                      'batch_graph_data': batch[1].to(args.device),
                      'ids': batch[2],
                      'in_train': False,
                      }
        else:
            batch.in_train = False
            inputs = batch.to(args.device)

        with torch.no_grad():
            pred = model(inputs, args)
            pred = pred.cpu().tolist()
            preds.extend(pred)
            ids.extend(batch.ids)
            if args.exp == "mol_pred":
                targets.extend(list(inputs.batch_graph_data.y.cpu().numpy()))
            else:
                targets.extend(inputs.labels.cpu().tolist())

            # print(preds, targets)

    preds = np.array(preds)
    if args.exp == "mol_pred":
        # print(targets, preds.tolist())
        score = roc_auc_score(targets, preds.tolist())
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        score2 = accuracy_score(targets, preds.tolist())

        args.logger.debug(f"accuracy_score {score2}")
        args.logger.debug(f"auc_score {score}", )
        # preds[preds >= 0.5] = 1
        # preds[preds < 0.5] = 0
        # score = accuracy_score(targets, preds.tolist())
        # args.logger.debug(f"accuracy_score {score}", )
        # # f1 = f1_score(targets, preds, average='macro')
    else:

        # print("targets, preds",targets, preds)
        # preds=preds.tolist()
        precision, recall, score2 = get_prf(targets, preds, average="macro", verbose=True)
        precision, recall, score = get_prf(targets, preds, average="micro", verbose=True)

        acc = accuracy_score(targets, preds)
        # print("targets",targets)
        # print("targets", len(targets), len(targets[0]))
        # print("preds",len(preds), len(preds[0]))

        args.logger.debug(f"acc {acc}", )
        # precision, recall, score = get_prf(targets, preds, average="samples")

    # output = None
    return score, [list(item) for item in zip(ids, preds, targets)]
