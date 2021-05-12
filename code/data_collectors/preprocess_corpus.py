import pandas as pd
from shutil import copyfile
import os
import glob
from utils import *
from sklearn.model_selection import train_test_split


"=========================="
# split a file to tvt

def split_to_tr_val_test(path, ratio="811"):
    f = load_file_lines(path)

    path_name, ext=get_path_name(path), get_ext(path)

    np.random.shuffle(f)
    mid1, mid2 = int(.8 * len(f)), int(.9 * len(f))

    dump_file(f[:mid1], path_name+"_train"+ext)
    dump_file(f[mid1:mid2], path_name+"_val"+ext)
    dump_file(f[mid2:], path_name+"_test"+ext)

fname="../data_online/chemet/test_jinfeng_b1.json"


split_to_tr_val_test("../data_online/chemet/test_jinfeng_b1.json", ratio="811")


"=========================="
#choose assigned



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
