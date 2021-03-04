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



def get_prf(targets, preds, average="micro", verbose=True):
    precision = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(targets, preds, average=average)
    print(precision, recall, f1)
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
            # texts = batch[0]
            # batch_ent1_d = batch[1]
            # batch_ent2_d = batch[3]
            #
            # # print("encoded_input", encoded_input)
            # inputs = {'texts': {key: texts[key].to(args.device) for key in texts},
            #           "batch_ent1_d": {key: batch_ent1_d[key].to(args.device) for key in batch_ent1_d},
            #           "batch_ent1_d_mask": batch[2].to(args.device),
            #           "batch_ent2_d": {key: batch_ent2_d[key].to(args.device) for key in batch_ent2_d},
            #           "batch_ent2_d_mask": batch[4].to(args.device),
            #           "batch_ent1_g": batch[5].to(args.device),
            #           "batch_ent1_g_mask": batch[6].to(args.device),
            #           "batch_ent2_g": batch[7].to(args.device),
            #           "batch_ent2_g_mask": batch[8].to(args.device),
            #           "ids": batch[9],
            #           "labels": batch[10].to(args.device),
            #           'in_train': False,
            #           }
            #
            batch.in_train=False
            inputs = batch.to(args.device)


        with torch.no_grad():
            pred = model(inputs, args)
            # print(pred.shape)
            # print(pred.cpu())
            # print(pred.cpu().numpy())
            # print(pred.cpu().numpy().squeeze(-1))
            pred = list(pred.cpu().numpy())
            preds.extend(pred)

            # print(inputs["ids"], inputs["batch_graph_data"].y, inputs["batch_graph_data"].y.cpu().numpy().squeeze(-1), list(inputs["batch_graph_data"].y.cpu().numpy().squeeze(-1)))
            # print("ergf", list(inputs["batch_graph_data"].y.cpu().numpy().squeeze()))

            if args.exp == "mol_pred":
                targets.extend(list(inputs["batch_graph_data"].y.cpu().numpy()))
            else:
                targets.extend(list(inputs["labels"].cpu().numpy()))
            # print("new targets", targets)

    # print("preds  ", preds)
    # print("targets", targets)
    # score = roc_auc_score(targets, preds)

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
        print("fff1", f1_score(targets, preds.tolist(), average="micro"))

        precision, recall, score=get_prf(targets, preds.tolist(), average="micro")

    # output = None
    return score, pred
