import pandas as pd
from shutil import copyfile
import os
import glob
from utils import *
from sklearn.model_selection import train_test_split
# from textacy.preprocessing.normalize import normalize_unicode
import re
from data_collectors.crawler_config import greek_alphabet

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
        # if tmp != t:
        #     print(t, "changed to", tmp)
        t = tmp
    return t


# mystring="hello 3\u03BF4 \u03C6 \u0391 \u03C9 23 23 ! $%&^ ^T "
# a=re.sub(r"[\x00-\x7f]+", "",mystring)
# a=list(a)
# a
# dump_file(a, "a.json")

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
