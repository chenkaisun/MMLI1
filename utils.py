# import torch
import sys
import re
import time
# import torch
from pprint import pprint as pp
import pickle as pkl
from collections import deque, defaultdict, Counter
import json
from IPython import embed
import random
import time
import os
import numpy as np
import argparse
import gc
from copy import deepcopy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data
# from pynvml import *
# from model.load_model import *
# from data import *
# from torch import optim
# from options import *
# import wandb




def join(str1, str2):
    return os.path.join(str1, str2)


def get_ext(filename):
    return os.path.splitext(filename)[1]
def dump_file(obj, filename):
    if get_ext(filename)==".json":
        with open(filename, "w+") as w:
            json.dump(obj, w)
    elif get_ext(filename)==".pkl":
        with open(filename, "wb+") as w:
            pkl.dump(obj, w)
    else:
        print("not pkl or json")
        with open(filename, "w+", encoding="utf-8") as w:
            w.write(obj)


def load_file(filename):
    if get_ext(filename)==".json":
        with open(filename, "r") as r:
            res = json.load(r)
    elif get_ext(filename) == ".pkl":
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res

def mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)