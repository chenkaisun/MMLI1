# import torch
# import torch
import os
import pickle as pkl
import json
# import time
# import os
import numpy as np
from bs4 import BeautifulSoup
# import argparse
# import gc
# from copy import deepcopy
# import torch.nn as nn
# import torch.nn.functional as F
# from pynvml import *
import requests
# from numba import jit
from matplotlib.pyplot import plot
import logging
from pynvml import *

logging.getLogger('matplotlib.font_manager').disabled = True

import matplotlib.pyplot as plt


def visualize_plot(x=None, y=None, name=None, path=""):
    for i, sub_y in enumerate(y):
        plt.plot(range(len(sub_y)) if not x else x[i], sub_y, 'o-', label=name[i])
    plt.legend()
    plt.savefig(path)
    # plt.show()


# def request_get(url, headers):
#     try:
#         r = requests.get(url, headers=headers)
#         if r.ok:
#             # print(r)
#             return r
#         else:
#
#             print(r)
#             return None
#     except Exception as e:
#         print(e)
#         return None
# from data import *
# import wandb
# def get_mole_desciption(r):
#     result = r.content
#     soup = BeautifulSoup(result, 'lxml')
#     sinple_item={}
#     for information in soup.find_all("information"):
#         CID = int(information.find('cid').get_text())
#         sinple_item["cid"]=CID
#         if information.find("title"):
#             sinple_item["title"]=information.find("title").get_text()
#
#         if information.find("description"):
#             if "descriptions" not in sinple_item:
#                 sinple_item["descriptions"] = []
#             description=information.find("description").get_text()
#             descriptionsourcename=information.find("descriptionsourcename").get_text()
#             descriptionurl=information.find("descriptionurl").get_text()
#             sinple_item["descriptions"].append({"description":description,
#                                                 "descriptionsourcename":descriptionsourcename,
#                                                 "descriptionurl":descriptionurl,})
#     return sinple_item

#
# r = request_get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{1}/description/XML')
# get_mole_desciption(r)


# @jit(nopython=True)
def is_symmetric(g):
    return np.sum(np.abs(g.T - g)) == 0


def join(str1, str2):
    return os.path.join(str1, str2)


def get_ext(filename):
    return os.path.splitext(filename)[1]


def get_path_name(filename):
    return os.path.splitext(filename)[0]


def dump_file(obj, filename):
    if get_ext(filename) == ".json":
        with open(filename, "w+") as w:
            json.dump(obj, w)
    elif get_ext(filename) == ".pkl":
        with open(filename, "wb+") as w:
            pkl.dump(obj, w)
    else:
        print("not pkl or json")
        with open(filename, "w+", encoding="utf-8") as w:
            w.write(obj)


def path_exists(path):
    return os.path.exists(path)


def load_file(filename, init=None):

    if not path_exists(filename) :

        if init is not None:
            print("file doesn't  exist, initializing")
            return init

    if get_ext(filename) == ".json":
        with open(filename, "r", encoding="utf-8") as r:
            res = json.load(r)


    elif get_ext(filename) == ".pkl":
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res


def load_file_default(filename, init=None):
    if not path_exists(filename):
        if init == "{}": return {}
        if init == "[]": return []
        if init == 0: return 0
        if init == "": return ""
        return None
    if get_ext(filename) == ".json":
        with open(filename, "r", encoding="utf-8") as r:
            res = json.load(r)
            # try:
            #     res = json.load(r)
            # except:
            #     print("here")
            #     res = [json.loads(line.strip()) for i, line in enumerate(r)]
            #     return res
            #     print(r)
    elif get_ext(filename) == ".pkl":
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res


def load_file_lines(filename):
    if get_ext(filename) == ".json":
        with open(filename, mode="r", encoding="utf-8") as fin:
            res = [json.loads(line.strip()) for i, line in enumerate(fin)]
    elif get_ext(filename) == ".pkl":
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res


def mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True

def get_gpu_mem_info():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')
