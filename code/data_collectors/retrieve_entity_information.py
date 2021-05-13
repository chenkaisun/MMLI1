# from pprint import pprint as pp
# import pickle as pkl
# from collections import deque, defaultdict, Counter
# import json
# from IPython import embed
# from pynvml import *
# import random
# import time
import os
import numpy as np
# import argparse
# import gc
# from copy import deepcopy
# import logging
# from data import load_data
import pubchempy as pcp
# from mediawiki import MediaWiki
import csv
from utils import dump_file, join, path_exists, load_file_default
from data_collectors.crawler_util import *
from bs4 import BeautifulSoup
import requests
import time


# from crawler_config import headers
# pcp.get_compounds("β-phenyldopamine", 'name')
# pcp.get_compounds("Methyl  phenylacetate", 'name')

def get_entity_info_chemprot():
    tr = "data_online/chemet/chemprot_training_entities.tsv"
    dev = "data_online/ChemProt_Corpus/chemprot_development/chemprot_development_entities.tsv"
    test = "data_online/ChemProt_Corpus/chemprot_test_gs/chemprot_test_entities_gs.tsv"
    data_dir = '../data_online/ChemProt_Corpus/'

    mention2ent = {}
    cmpd_info = {}

    batch_save = 100
    cnt = 0
    for file in [tr, dev, test]:
        with open(file, encoding='utf-8') as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for i, row in enumerate(rd):
                # if i<3: continue

                print(i, row)
                if not row or row[2] != "CHEMICAL": continue

                mole_name = row[-1]

                if mole_name not in mention2ent:
                    try:
                        results = pcp.get_compounds(mole_name, 'name')
                    except Exception as e:
                        print(e)
                        time.sleep(8)
                        continue
                    # r = request_get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/water/cids/TXT')
                    # print(r.text)

                    # if r and r.ok:
                    #     desc = get_mole_desciption(r)
                    #     print(desc)

                    # if no match, do not repeat search for the same mention
                    mention2ent[mole_name] = None

                    # if there is a match
                    if results:
                        cmpd = results[0]
                        print("Has results", cmpd)

                        cid = cmpd.cid
                        mention2ent[mole_name] = cid

                        # create compound entry
                        if cid not in cmpd_info:
                            cmpd_info[cid] = cmpd.to_dict()
                            # cmpd_info[cid]["canonical_smiles"]=cmpd.canonical_smiles
                            # cmpd_info[cid]["fingerprint"]=cmpd.fingerprint
                            # cmpd_info[cid]["exact_mass"]=cmpd.exact_mass
                            # cmpd_info[cid]["h_bond_acceptor_count"]=cmpd.h_bond_acceptor_count
                            # cmpd_info[cid]["molecular_weight"]=cmpd.molecular_weight
                            # cmpd_info[cid]["synonyms"]=cmpd.synonyms
                            # cmpd_info[cid]["charge"]=cmpd.charge
                            # cmpd_info[cid]["iupac_name"]=cmpd.iupac_name
                            print("here")

                            r = request_get(
                                f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(cid)}/description/XML')
                            if r and r.ok:
                                desc = get_mole_desciption(r)
                                print("desc", desc)

                                cmpd_info[cid]["pubchem_description"] = desc
                                # print("cmpd_info", cmpd_info)
                            # wikihtml_file = 'data/wikipedia_html_for_pubchem_compounds/' + str(cid) + '.html'
                            #
                            # print(wikihtml_file)
                            # if os.path.exists(wikihtml_file):
                            #     print("Exist html")
                            #     with open(wikihtml_file, encoding='utf-8') as htmld:
                            #         soup = BeautifulSoup(htmld, 'html.parser')
                            #         infobox = soup.find('table.infobox.bordered')
                            #         print(infobox.next_sibling)
                            #         print(infobox.next_sibling.next_sibling)
                            # while infobox.find_next('p').next_sibling:
                            # for a in description.findAll('a'):
                            #     a.replaceWithChildren()
                            #
                            # print(description)
                            # print(description.get_text())
                            # cmpd_info["wiki_description"]=description.get_text()

                    cnt += 1
                    if cnt % batch_save == 0:
                        dump_file(mention2ent, data_dir + "mention2ent.json")
                        dump_file(cmpd_info, data_dir + "cmpd_info.json")
            dump_file(mention2ent, data_dir + "mention2ent.json")
            dump_file(cmpd_info, data_dir + "cmpd_info.json")


data_dir = '../data_online/chemet/'


def get_entity_info_fet(files=None):
    # tr = "data_online/chemet/chemprot_training_entities.tsv"
    # dev = "data_online/ChemProt_Corpus/chemprot_development/chemprot_development_entities.tsv"
    # test = "data_online/ChemProt_Corpus/chemprot_test_gs/chemprot_test_entities_gs.tsv"

    tmp = []
    dump_file(tmp, data_dir + "tmp_empty.json")

    mention2ent_path = data_dir + "mention2ent.json"
    mention2ent = load_file(mention2ent_path, init={})
    cmpd_info_path = data_dir + "cmpd_info.json"
    cmpd_info = load_file(cmpd_info_path, init={})

    print("init", mention2ent, cmpd_info)

    batch_save = 100
    cnt = 0
    n_match = 0
    for file in files:

        for i, sample in enumerate(file):
            tokens = sample["tokens"]
            for j, m in enumerate(sample["annotations"]):
                s, e = m["start"], m["end"]
                mole_name = " ".join(tokens[s:e])
                mole_name = mole_name.replace("  ", " ")

                cnt += 1

                if mole_name not in mention2ent:

                    # get possible substrings
                    possible_mole_names = [mole_name]
                    if e - s > 1:
                        possible_mole_names.append(" ".join(tokens[s:e - 1]))
                        possible_mole_names.append(" ".join(tokens[s + 1:e]))
                    if e - s > 2:
                        possible_mole_names.append(" ".join(tokens[s:e - 2]))
                        possible_mole_names.append(" ".join(tokens[s + 2:e]))
                    results = None
                    try:
                        for possible_name in possible_mole_names:
                            results = pcp.get_compounds(possible_name, 'name')
                            if results:
                                break

                    except Exception as e:
                        print(e)
                        time.sleep(8)
                        continue
                    # r = request_get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/water/cids/TXT')
                    # print(r.text)

                    # if r and r.ok:
                    #     desc = get_mole_desciption(r)
                    #     print(desc)

                    ## if no match, do not repeat search for the same mention
                    mention2ent[mole_name] = None

                    # if there is a match
                    if results:
                        cmpd = results[0]

                        print("\nHas results", mole_name, cmpd)
                        n_match += 1  # n entities having match

                        cid = cmpd.cid
                        cid = str(cid)
                        mention2ent[mole_name] = cid

                        # create compound entry
                        if cid not in cmpd_info:
                            cmpd_info[cid] = cmpd.to_dict()
                            # print("here")

                            r = request_get(
                                f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(cid)}/description/XML',
                                headers)
                            if r and r.ok:
                                desc = get_mole_desciption(r)
                                print("desc", desc)
                                cmpd_info[cid]["pubchem_description"] = desc
                        else:
                            print(f"{cid} already in info")
                    if cnt % batch_save == 0:
                        dump_file(mention2ent, data_dir + "mention2ent.json")
                        dump_file(cmpd_info, data_dir + "cmpd_info.json")

            dump_file(mention2ent, data_dir + "mention2ent.json")
            dump_file(cmpd_info, data_dir + "cmpd_info.json")

    print("cnt", cnt)
    print("n_match (distinct mentions)", n_match)

    #
    # n_match = 0
    # with open(file, encoding='utf-8') as fd:
    #     rd = csv.reader(fd, delimiter="\t", quotechar='"')
    #     for i, row in enumerate(rd):
    #         # if i<3: continue
    #
    #         print(i, row)
    #         if not row or row[2] != "CHEMICAL": continue
    #
    #         mole_name = row[-1]
    #
    #         if mole_name not in mention2ent:
    #             try:
    #                 results = pcp.get_compounds(mole_name, 'name')
    #             except Exception as e:
    #                 print(e)
    #                 time.sleep(8)
    #                 continue
    #             # r = request_get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/water/cids/TXT')
    #             # print(r.text)
    #
    #             # if r and r.ok:
    #             #     desc = get_mole_desciption(r)
    #             #     print(desc)
    #
    #             ## if no match, do not repeat search for the same mention
    #             mention2ent[mole_name] = None
    #
    #             # if there is a match
    #             if results:
    #                 cmpd = results[0]
    #                 print("Has results", cmpd)
    #
    #                 cid = cmpd.cid
    #                 mention2ent[mole_name] = cid
    #                 n_match+=1
    #
    #                 # create compound entry
    #                 if cid not in cmpd_info:
    #                     cmpd_info[cid] = cmpd.to_dict()
    #                     print("here")
    #
    #                     r = request_get(
    #                         f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(cid)}/description/XML', headers)
    #                     if r and r.ok:
    #                         desc = get_mole_desciption(r)
    #                         print("desc", desc)
    #
    #                         cmpd_info[cid]["pubchem_description"] = desc
    #
    #             cnt += 1
    #             if cnt % batch_save == 0:
    #                 dump_file(mention2ent, data_dir + "mention2ent.json")
    #                 dump_file(cmpd_info, data_dir + "cmpd_info.json")
    #     dump_file(mention2ent, data_dir + "mention2ent.json")
    #     dump_file(cmpd_info, data_dir + "cmpd_info.json")
    #     print(tr, "n_match",n_match)


fname1 = data_dir + "test_jinfeng_b_cleaned.json"
fname2 = data_dir + "test_chem_anno_cleaned.json"
# fname = outfname
# f = load_file_lines(fname)
f1 = load_file(fname1)
f2 = load_file(fname2)
get_entity_info_fet([f1, f2])
