import numpy as np
import pandas as pd
from shutil import copyfile
import os
import glob
from utils import *
from sklearn.model_selection import train_test_split
# from textacy.preprocessing.normalize import normalize_unicode
import re
from data_collectors.crawler_config import greek_alphabet
from collections import defaultdict
from copy import deepcopy
"=========================="


# split a file to tvt

def split_to_tr_val_test(path, ratio="811"):
    # f = load_file_lines(path)
    f = load_file_lines(path)

    path_name, ext = get_path_name(path), get_ext(path)

    np.random.shuffle(f)
    mid1, mid2 = int(.8 * len(f)), int(.9 * len(f))

    dump_file(f[:mid1], path_name + "_train" + ext)
    dump_file(f[mid1:mid2], path_name + "_val" + ext)
    dump_file(f[mid2:], path_name + "_test" + ext)


# def get_all_unicode(path):
#     f = load_file_lines(path)
#     u_code=set()
#     for sample in f:
#         for t in sample["tokens"]:
#             if "\\u" in t:
#                 u_code.add(t)
#     return u_code
def get_all_unicode(sent):
    return re.sub(r"[\x00-\x7f]+", "", sent)


def clean_text(t):
    for a in greek_alphabet:
        tmp = t.replace(a, greek_alphabet[a])
        t = tmp
    return t


"""=========checking dataset right========="""
fname_tr = "../data_online/chemet/distant_training.json"
fname_dev = "../data_online/chemet/dev_anno.json"
fname_te = "../data_online/chemet/test_anno.json"
outfname_tr = "../data_online/chemet/distant_training_new.json"
outfname_dev = "../data_online/chemet/dev_anno_unseen_removed.json"
outfname_te = "../data_online/chemet/test_anno_unseen_removed.json"
test_jinfeng_b = "../data_online/chemet/test_jinfeng_b.json"
# test_jinfeng_b = "../data_online/chemet/test_jinfeng_b.json"
# test_jinfeng_b = "../data_online/chemet/test_jinfeng_b.json"


f_tr = load_file_lines(fname_tr)
f_dev = load_file_lines(fname_dev)
f_te = load_file_lines(fname_te)
f_jf = load_file_lines(test_jinfeng_b)

docs_te = set([sample["doc_id"] for f in [f_dev, f_te] for i, sample in enumerate(f)])
docs_tr = set([sample["doc_id"] for i, sample in enumerate(f_tr)])
docs_jf = set([sample["doc_id"] for i, sample in enumerate(f_jf)])

####see te doc dif from tr
print(docs_te.intersection(docs_tr))

###same as dist
print(docs_jf.difference(docs_tr))
print(docs_tr.difference(docs_jf))

len(docs_tr)
len(docs_te)
len(docs_te)

len(f_tr)
len(f_dev)
len(f_te)


# f_tr = [sample for sample in f_tr if sample["doc_id"] not in docs_te]
"""========================="""

def get_label_occurrence(f):
    labels_tr = defaultdict(int)

    for i, sample in enumerate(f):
        for m in sample["annotations"]:
            for lb in m["labels"]:
                labels_tr[lb] += 1
            # labels_tr = labels_tr.union(m["labels"])
            assert len(m["labels"]) > 0
    return labels_tr


a, b, c = get_label_occurrence(f_tr), get_label_occurrence(f_dev), get_label_occurrence(f_te)

ltr = set(get_label_occurrence(f_tr).keys())
ldev = set(get_label_occurrence(f_dev).keys())
lte = set(get_label_occurrence(f_te).keys())
print("e0")
print(len(ldev.intersection(ltr)))
print(len(lte.intersection(ltr)))
print(len(ldev.difference(ltr)))
print(len(lte.difference(ltr)))

list(a.values()).sort()
print(np.array().sort())

for k in b:
    a[k] += b[k]
for k in b:
    a[k] += b[k]

print("ltr", ltr)
print("ldev", ldev)
print("lte", lte)

for k, d in enumerate([f_dev, f_te]):
    for i, sample in enumerate(d):
        tmp_a = deepcopy(d[i]["annotations"])
        d[i]["annotations"] = []
        for j, m in enumerate(tmp_a):
            tmp_a[j]["labels"] = list(set(m["labels"]).intersection(ltr))
            if len(tmp_a[j]["labels"]):
                d[i]["annotations"].append(tmp_a[j])
f_dev=[s for s in f_dev if len(s["annotations"])>0]
f_te=[s for s in f_te if len(s["annotations"])>0]


# for i, sample in enumerate(f):
#     for j, t in enumerate(sample["tokens"]):
#         sample["tokens"][j] = clean_text(t)
dump_file(f_tr, outfname_tr)
dump_file(f_dev, outfname_dev)
dump_file(f_te, outfname_te)

# labels_tr = defaultdict(int)
# for i, sample in enumerate(f_tr):
#     for m in sample["annotations"]:
#
#         for lb in m["labels"]:
#             labels_tr[lb]+=1
#         # labels_tr = labels_tr.union(m["labels"])
#         assert len(m["labels"])>0


# print("labels_tr", labels_tr)


# for i, sample in enumerate(f_te):
#     for j, m in enumerate(sample["annotations"]):
#
#         f_te[i]["annotations"][j]["labels"] = list(set(m["labels"]).intersection(labels_tr))

# mystring="hello 3\u03BF4 \u03C6 \u0391 \u03C9 23 23 ! $%&^ ^T "
# a=re.sub(r"[\x00-\x7f]+", "",mystring)
# a=list(a)
# a
# dump_file(a, "a.json")


f_tr = load_file("../data_online/chemet/distant_training_new.json")
f_dev = load_file("../data_online/chemet/dev_anno_unseen_removed.json")
f_te = load_file("../data_online/chemet/test_anno_unseen_removed.json")

print(sum([any([len(m['labels'])==0 for m in s["annotations"]]) for s in f_te]))
print(sum([any([len(m['labels'])==0 for m in s["annotations"]]) for s in f_dev]))
print(sum([len(s["annotations"])==0 for s in f_dev]))

f_jf = load_file_lines(test_jinfeng_b)

"""=========remove docs from tr that appeared in test data and clean test labels that contain unseen label========="""

fname_tr = "../data_online/chemet/test_jinfeng_b_cleaned.json"
fname_te = "../data_online/chemet/test_chem_anno_cleaned.json"
out_fname_tr = "../data_online/chemet/test_jinfeng_b_cleaned_cleaned.json"
out_fname_te = "../data_online/chemet/test_chem_anno_cleaned_cleaned.json"

f_tr = load_file(fname_tr)
f_te = load_file(fname_te)

docs_te = set([sample["doc_id"] for i, sample in enumerate(f_te)])
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

labels_te = set()
for i, sample in enumerate(f_te):
    for m in sample["annotations"]:
        labels_te = labels_te.union(m["labels"])
print("labels_te", labels_te)
print(labels_te.difference(labels_tr))
print(labels_tr.difference(labels_te))

"""=========clean files========="""
# clean chem annos
fname = "../data_online/chemet/test_jinfeng_b.json"
outfname = "../data_online/chemet/test_jinfeng_b_cleaned.json"
f = load_file_lines(fname)
# fname = outfname
# fname = "../data_online/chemet/test_chem_anno.json"
# outfname = "../data_online/chemet/test_chem_anno_cleaned.json"
# f = load_file(fname)

cnt = 0
for i, sample in enumerate(f):
    for j, t in enumerate(sample["tokens"]):
        sample["tokens"][j] = clean_text(t)
dump_file(f, outfname)

# ####tmp
# fname = "../data_online/chemet/test_chem_anno_cleaned_cleaned.json"
# f = load_file(fname)
# # fname = outfname
# # fname = "../data_online/chemet/test_chem_anno.json"
# # outfname = "../data_online/chemet/test_chem_anno_cleaned.json"
# # f = load_file(fname)
#
# cnt = 0
# for i, sample in enumerate(f):
#     for j, t in enumerate(sample["tokens"]):
#         sample["tokens"][j] = clean_text(t)
# dump_file(f, outfname)
# #####

# retrieval

exit()

# split
fname = "../data_online/chemet/test_jinfeng_b1.json"
fname = "../data_online/chemet/test_jinfeng_b.json"
split_to_tr_val_test(fname, ratio="811")

# change
fname = "../data_online/chemet/test_chem_anno.json"
ofname = "../data_online/chemet/test_chem_anno.json"

dump_file(load_file_lines(fname), fname)

"=========================="
# choose assigned


dir = "../other_repos\oscar4-master\oscar4-tokeniser\chem_anno\\"
new_dir = "../other_repos/oscar4-master/oscar4-tokeniser/chem_anno_selected/"
token_dir = "../other_repos/oscar4-master/oscar4-tokeniser/chem_anno_tokenized/"

df = pd.read_csv(new_dir + "AnnotationAssignments.csv", header=None)
# print(df[1])
for fname in list(df[1]):
    name = fname.split("/")[-1]
    print(name)
    copyfile(dir + name + ".txt", new_dir + name + ".txt")

    name = name.replace("_art", "_abs") if "_art" in name else name.replace("_abs", "_art")
    print(name)

    copyfile(dir + name + ".txt", new_dir + name + ".txt")

print(list(df))
for row in df:
    print(row)

txtfiles = []
cnt = 0
# print(glob.glob(token_dir+"*"))
for file in glob.glob(token_dir + "*"):
    with open(file, encoding="utf-8") as rf:
        for i, line in enumerate(rf):
            # print(file, i, line)
            cnt += 1
            if len(line.strip().split("\n")) > 1:
                print(file, i, line)
print(cnt)
