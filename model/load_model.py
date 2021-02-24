from torch import optim
import torch
import os
from model.joint_gt import JNet

from model.re_model import RE


def get_model(args):
    if args.model_name=="joint_gt":
        return JNet(args)
    if args.model_name=="re_model":
        return RE(args)



def load_model_from_path(model, optimizer, model_path, gpu=True):
    model_epoch, best_dev_score = 0, -float("inf")

    if False:#os.path.isfile(model_path):
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
        if gpu: model = model.cuda()

    return model, optimizer, model_epoch, best_dev_score
