from utils import *
from data_collectors.crawler_config import *
import pubchempy as pcp
# from mediawiki import MediaWiki
import csv
from bs4 import BeautifulSoup
import requests
import time
import sys
import json
from data_collectors.crawler_config import *
tr = "data_online/ChemProt_Corpus/chemprot_training/chemprot_training_entities.tsv"
dev = "data_online/ChemProt_Corpus/chemprot_development/chemprot_development_entities.tsv"
test = "data_online/ChemProt_Corpus/chemprot_test_gs/chemprot_test_entities_gs.tsv"
data_dir = 'data_online/ChemProt_Corpus/'

# mention2ent = {}
# cmpd_info = {}
mention2protid = {} # load_file(data_dir+"mention2ent.json")
prot_info = {}

mention2protid =  load_file(data_dir+"mention2protid.json")
prot_info =load_file(data_dir+"prot_info.json")

out1="mention2protid.json"
out2="prot_info.json"

batch_save = 3
cnt = 0

#
# alphabet_map={
#     chr(945):"Alpha",
#     chr(946):"Beta",
#     chr(947):"Gamma",
#     chr(948):"Delta",
#     chr(949):"Epsilon",
#     chr(950):"Zeta",
#
#
# }
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
}
headers["Accept"]="application/json"
for file in [tr, dev, test]:
    with open(file, encoding='utf-8') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):

            print(i, row)
            # if i<3: continue
            # if i==9592:
            #     prot_name = row[-1]
            #     print(i, row)
            #
            #     tmp=prot_name
            #     for a in greek_alphabet:
            #         tmp=tmp.replace(a, greek_alphabet[a])
            #     print(tmp)
                # print(prot_name[0] in greek_alphabet)
                #
                #
                # print(prot_name.replace(chr(945),"alpha"))
                # print(prot_name.replace(chr(945),"alpha"))
                # print(chr(945) in prot_name)
                # print(prot_name[0]==chr(945))
                # "\u0391".encode("utf-8")
                # tmp = unidecode(prot_name[0].decode("utf-8"))
            #     print(prot_name[0])
            #
            # continue
            if not row or "GENE-Y" not in row[2]: continue

            prot_name = row[-1]


            if prot_name not in mention2protid:

                tmp = prot_name


                for a in greek_alphabet:
                    tmp=tmp.replace(a, greek_alphabet[a])
                print("tmp", tmp)

                r=request_get(f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/search?query={tmp}&limit=1&page=1", headers)
                #
                # try:
                #
                #
                #
                #     requestURL =
                #     r = requests.get(requestURL, headers=headers)
                #
                #     if not r.ok:
                #         r.raise_for_status()
                #         sys.exit()
                #
                #     responseBody = r.text
                #     print(responseBody)
                # except Exception as e:
                #     print(e)
                #     time.sleep(3)
                #     continue

                mention2protid[prot_name] = None


                if r:

                    results=json.loads(r.text)["results"]
                    # print(results)

                    # if there is a match
                    if results and 'id' in results[0]:
                        results=results[0]
                        pid=results['id']
                        mention2protid[prot_name]=pid
                        if pid not in prot_info:
                            prot_info[pid] = results
                        print(results)
                        # print("here")
                        # print(prot_info)
                    cnt += 1
                    if cnt % batch_save == 0:
                        # print("dumping")
                        # print(data_dir + out2)
                        # print(prot_info)
                        dump_file(mention2protid, data_dir + out1)
                        dump_file(prot_info, data_dir + out2)
        dump_file(mention2protid, data_dir + out1)
        dump_file(prot_info, data_dir + out2)