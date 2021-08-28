import pandas as pd
from shutil import copyfile
import os
import glob
from sklearn.model_selection import train_test_split
# from textacy.preprocessing.normalize import normalize_unicode
import re
from utils import *

greek_alphabet = {
    u'\u0391': 'Alpha',
    u'\u0392': 'Beta',
    u'\u0393': 'Gamma',
    u'\u0394': 'Delta',
    u'\u0395': 'Epsilon',
    u'\u0396': 'Zeta',
    u'\u0397': 'Eta',
    u'\u0398': 'Theta',
    u'\u0399': 'Iota',
    u'\u039A': 'Kappa',
    u'\u039B': 'Lamda',
    u'\u039C': 'Mu',
    u'\u039D': 'Nu',
    u'\u039E': 'Xi',
    u'\u039F': 'Omicron',
    u'\u03A0': 'Pi',
    u'\u03A1': 'Rho',
    u'\u03A3': 'Sigma',
    u'\u03A4': 'Tau',
    u'\u03A5': 'Upsilon',
    u'\u03A6': 'Phi',
    u'\u03A7': 'Chi',
    u'\u03A8': 'Psi',
    u'\u03A9': 'Omega',
    u'\u03B1': 'alpha',
    u'\u03B2': 'beta',
    u'\u03B3': 'gamma',
    u'\u03B4': 'delta',
    u'\u03B5': 'epsilon',
    u'\u03B6': 'zeta',
    u'\u03B7': 'eta',
    u'\u03B8': 'theta',
    u'\u03B9': 'iota',
    u'\u03BA': 'kappa',
    u'\u03BB': 'lamda',
    u'\u03BC': 'mu',
    u'\u03BD': 'nu',
    u'\u03BE': 'xi',
    u'\u03BF': 'omicron',
    u'\u03C0': 'pi',
    u'\u03C1': 'rho',
    u'\u03C3': 'sigma',
    u'\u03C4': 'tau',
    u'\u03C5': 'upsilon',
    u'\u03C6': 'phi',
    u'\u03C7': 'chi',
    u'\u03C8': 'psi',
    u'\u03C9': 'omega',

    u'\u2013': '-',
    u'\u2212': '-',
    # u'\u25b3': 'change of ',
    u'\u2018': '\'',
    u'\u2019': '\'',
    u'\u201c': '\"',
    u'\u201d': '\"',
    u'\u2032': '\'',  # prime
    u'\u00b0C': " degrees Celsius",
    u'\u00b0F': " degrees Fahrenheit"

}


def clean_text(t):
    for a in greek_alphabet:
        tmp = t.replace(a, greek_alphabet[a])
        # if tmp != t:
        #     print(t, "changed to", tmp)
        t = tmp
    return t


def clean_files(train_file, val_file, test_file):
    """"""

    """=========remove docs from tr that appeared in test data and clean test labels that contain unseen label========="""
    f_tr = load_file_lines(train_file)
    f_val = load_file_lines(val_file)
    f_te = load_file_lines(test_file)

    # this is for double check
    docs_te = set([sample["doc_id"] for f in [f_val,f_te] for i, sample in enumerate(f)])
    f_tr = [sample for sample in f_tr if sample["doc_id"] not in docs_te]

    labels_tr = set()
    for i, sample in enumerate(f_tr):
        for m in sample["annotations"]:
            labels_tr = labels_tr.union(m["labels"])
    print("labels_tr", labels_tr)

    for i, sample in enumerate(f_te):
        for j, m in enumerate(sample["annotations"]):
            f_te[i]["annotations"][j]["labels"] = list(set(m["labels"]).intersection(labels_tr))

    # for i, sample in enumerate(f):
    #     for j, t in enumerate(sample["tokens"]):
    #         sample["tokens"][j] = clean_text(t)
    dump_file(f_tr, out_fname_tr)
    dump_file(f_te, out_fname_te)

    pass
