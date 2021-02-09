from pprint import pprint as pp
import pickle as pkl
from collections import deque, defaultdict, Counter
import json
from IPython import embed
from pynvml import *
import random
import time
import os
import numpy as np
import argparse
import gc
from mediawiki import MediaWiki

import urllib
from urllib.request import urlopen
from urllib.parse import urljoin
import requests
import pubchempy as pcp
from bs4 import BeautifulSoup
# from utils import *
import lxml


def dump_file(obj, filename, type="pkl"):
    if type == "json":
        with open(filename, "w+") as w:
            json.dump(obj, w)
    elif type == "pkl":
        with open(filename, "wb+") as w:
            pkl.dump(obj, w)
    else:
        print("not pkl or json")
        with open(filename, "w+", encoding="utf-8") as w:
            w.write(obj)

def load_file(filename, type="pkl"):
    if type != "pkl":
        with open(filename, "r") as r:
            res = json.load(r)
    else:
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res

# with open("compound_info - Copy.json", 'r', encoding='utf-8') as r:
#     soup = BeautifulSoup(r, 'lxml')
#     for i in soup.find_all('tr'):
#         if i.find('p'):
#             properties.append(i.find('p').get_text().strip())
def update_record(items, r,parent_name,  tags):
    result = r.content
    # dump_file(r.text, 'tmp.xml', 'other')
    soup = BeautifulSoup(result, 'lxml')
    # print(soup)
    # print(soup.find_all(parent_name))
    # embed()
    for information in soup.find_all(parent_name):
        CID = int(information.find('cid').get_text())
        for tag in tags:

            tag_content = information.find(tag)
            if tag_content:
                items[CID][tag] = tag_content.get_text()
    return items


headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
wikipedia = MediaWiki(
    user_agent='Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6')

start_id = 1
num_items = 5000
chunk_size = 200
items = [{} for _ in range(num_items + 1)]

properties = ['MolecularFormula', 'MolecularWeight', 'CanonicalSMILES', 'IsomericSMILES', 'InChI', 'InChIKey',
              'IUPACName', 'XLogP', 'ExactMass', 'MonoisotopicMass', 'TPSA', 'Complexity', 'Charge', 'HBondDonorCount',
              'HBondAcceptorCount', 'RotatableBondCount', 'HeavyAtomCount', 'IsotopeAtomCount', 'AtomStereoCount',
              'DefinedAtomStereoCount', 'UndefinedAtomStereoCount', 'BondStereoCount', 'DefinedBondStereoCount',
              'UndefinedBondStereoCount', 'CovalentUnitCount', 'Volume3D', 'XStericQuadrupole3D', 'YStericQuadrupole3D',
              'ZStericQuadrupole3D', 'FeatureCount3D', 'FeatureAcceptorCount3D', 'FeatureDonorCount3D',
              'FeatureAnionCount3D', 'FeatureCationCount3D', 'FeatureRingCount3D', 'FeatureHydrophobeCount3D',
              'ConformerModelRMSD3D', 'EffectiveRotorCount3D', 'ConformerCount3D', 'Fingerprint2D']
properties_str=','.join(properties)
#
# r = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{1}/property/{properties_str}/XML',
#                  headers=headers)
# if r.ok:
#     print("ok", r.content)
#     pp(r)
#     items = update_record(items, r, 'properties', [tag.lower() for tag in properties])
#     print(items)
if not os.path.isdir("wikipedia_html"): os.mkdir("wikipedia_html")
chunk_pos=0
for i in range(start_id, num_items + start_id, chunk_size):
    c_range = ",".join([str(j) for j in range(i, min(i + chunk_size, num_items + start_id))])
    r = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{c_range}/property/{properties_str}/XML',
                     headers=headers)
    if r.ok:
        # print("ok", r.content)

        items = update_record(items, r, 'properties', [tag.lower() for tag in properties])
        # print(items)
    r = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{c_range}/description/XML',
                     headers=headers)
    if r.ok:
        # print("ok", r.content)
        items = update_record(items, r,'information',
                                   ['title', 'description', 'descriptionsourcename', 'descriptionurl', 'description'])
        # print(items)
    dump_file(items, "compound_info.json", 'json')
    print("iter", chunk_pos)
    chunk_pos+=1

    # wiki pages
    for j in range(i,min(i + chunk_size, num_items + start_id)):
        if 'title' in items[j]:
            try:
                print(items[j]['title'])
                if len(items[j]['title'])<255 and "CID " not in items[j]['title']:
                    res=wikipedia.prefixsearch(items[j]['title'])
                    if len(res):
                        p = wikipedia.page(res[0])
                        # items[j]['html']=p.html
                        # print(p.html)
                        dump_file(p.html, f"wikipedia_html/{j}.html", 'html')
            except:
                pass
exit()

wikipedia = MediaWiki(
    user_agent='Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6')

res=wikipedia.prefixsearch('ace')
p = wikipedia.page(res[0])
# items[j]['html']=p.html
print(p.html)
with open(f"wikipedia_html/temp.html", "w+", encoding="utf-8") as w:
    w.write(p.html)
dump_file(p.html, f"wikipedia_html/temp.html", 'html')
exit()

c_names = []
for i in range(1, 5000):
    r = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{i}/description/XML', headers=headers)
    if r.ok:
        html = r.content
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title').get_text()
        summary = soup.find('div', attrs={"class": 'summary p-md-top p-md-bottom'})

        pp(summary)
        desc = summary.find_all('th', attrs={"class": 'limited align-left'}).get_text()
        pp(desc)

        if " | " in title:
            c_name = title.split("|")[0].strip()
            if "CID " not in c_name:
                c_names.append(c_name)
                dump_file(c_names, "c_names.json", 'json')

for i in range(10):
    wikipedia.prefixsearch(c_names)

    p = wikipedia.page('Acetyl L Carnitine')
    print(p.html)

exit()
wikipedia.search('washington')
wikipedia.prefixsearch('wefewashington')
p = wikipedia.page('Martha Washington')
pp(p.content)

with open('ewf.html', 'w+', encoding='utf-8') as f:
    f.write(p.html)
pp(p.wikitext)
p.section('U.S. paper currency')

import pubchempy as pcp

for i in range(1, 100):
    c = pcp.Compound.from_cid(i)
    print(dir(c))
    p = pcp.get_properties('IsomericSMILES', 'CC', 'smiles', searchtype='superstructure')

    pcp.Compound.from_cid(962)
    dir(c)
    print(c.shape_selfoverlap_3d)

c = pcp.Compound.from_cid(5090)
c.iupac_name
