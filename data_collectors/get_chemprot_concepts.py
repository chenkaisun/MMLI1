from utils import join, dump_file
import pubchempy as pcp
# from mediawiki import MediaWiki
import csv
from bs4 import BeautifulSoup
import requests
import time
import sys
import json
from data_collectors.crawler_config import *
from data_collectors.crawler_util import *
from IPython import embed

"""directly run to collect protein data """
data_dir = '../data_online/ChemProt_Corpus/'
tr = join(data_dir, "chemprot_training/chemprot_training_entities.tsv")
dev = join(data_dir, "chemprot_development/chemprot_development_entities.tsv")
test = join(data_dir, "chemprot_test_gs/chemprot_test_entities_gs.tsv")

# headers["Accept"] = "application/json"
url = 'https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance={}&topK=20'

mention2concepts = {}
for file in [tr, dev, test]:
    with open(file, encoding='utf-8') as fd:
        rd = list(csv.reader(fd, delimiter="\t", quotechar='"'))
        rd_len = sum(1 for row in rd)

        for i, row in enumerate(rd):

            if not row: continue

            mention = row[-1]

            if mention not in mention2concepts:
                tmp = mention

                for a in greek_alphabet:
                    tmp = tmp.replace(a, greek_alphabet[a])
                print("tmp", tmp)

                r = request(url.format(tmp), headers)
                if r:
                    mention2concepts[mention] = get_concepts(r)
                    print(mention2concepts)
            if not i % 100 or i == rd_len - 1:
                dump_file(mention2concepts, join(data_dir, "mention2concepts.json"))
