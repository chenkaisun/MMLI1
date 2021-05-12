import pandas as pd
from shutil import copyfile
import os
import glob
from utils import *






dir="../other_repos\oscar4-master\oscar4-tokeniser\chem_anno\\"
new_dir="../other_repos/oscar4-master/oscar4-tokeniser/chem_anno_selected/"
token_dir="../other_repos/oscar4-master/oscar4-tokeniser/chem_anno_tokenized/"

df=pd.read_csv(new_dir+"AnnotationAssignments.csv", header=None)
# print(df[1])
for fname in list(df[1]):
    name=fname.split("/")[-1]
    print(name)
    copyfile(dir+name+".txt", new_dir+name+".txt")

    name=name.replace("_art", "_abs") if "_art" in name else name.replace("_abs", "_art")
    print(name)

    copyfile(dir+name+".txt", new_dir+name+".txt")

print(list(df))
for row in df:
    print(row)



txtfiles = []
cnt=0
# print(glob.glob(token_dir+"*"))
for file in glob.glob(token_dir+"*"):
    with open(file, encoding="utf-8") as rf:
        for i,line in enumerate(rf):
            # print(file, i, line)
            cnt+=1
            if len(line.strip().split("\n"))>1:
                print(file, i, line)
print(cnt)