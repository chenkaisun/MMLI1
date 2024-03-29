# from torch import optim
import torch
from torch import nn
# import os
from model.joint_gt import JNet

from model.re_model import RE
from model.fet import FET
from model.gnn import *
from model.baseline import *

def get_model(args, **kwargs):
    if args.model_name == "joint_gt":
        # return MoleGNN2(args)
        return JNet(args)
    elif args.model_name == "re_model":
        return RE(args)

    elif args.model_name == "fet_model":
        return FET(args)

    elif args.model_name == "lstm":
        return LstmFet(args, kwargs['pretrained_weights'])
def load_model_from_path(model, optimizer, args):
    model_epoch, best_dev_score = 0, -float("inf")

    if False:  # os.path.isfile(model_path):
        print("Saved model found")
        saved_model_info = torch.load(model_path)
        model.load_state_dict(saved_model_info['model_state_dict'])
        model_epoch = saved_model_info["epoch"]
        best_dev_score = saved_model_info["best_val_score"]
        optimizer.load_state_dict(saved_model_info['optimizer_state_dict'])
        if gpu:
            model = model.cuda()
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
    else:
        if str(args.device) != "cpu":
            print("use gpu")
            # if torch.cuda.device_count() > 1:
            #     print("parallel")
            #     model = nn.DataParallel(model)
            model = model.to(args.device)

    return model, optimizer, model_epoch, best_dev_score
