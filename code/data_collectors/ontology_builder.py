# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/

from utils import *
import sys
from SPARQLWrapper import SPARQLWrapper, JSON

import sys
import os
from pprint import pprint as pp
import pickle as pkl
from collections import deque, defaultdict, Counter
import json
from pynvml import *
import random
import time
import os
import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--npages', default=1000000, type=int)
args = parser.parse_args()

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/39.0.2171.95 Safari/537.36'}

endpoint_url = "https://query.wikidata.org/sparql"


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setTimeout(3000)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def get_rel_id():
    query = """SELECT DISTINCT ?rel
    WHERE
    {
     ?item wdt:P31 wd:Q11173.
    #   ?item wdt:P662 ?value.
      ?item ?rel ?obj.

      ?article schema:about ?item ;
          schema:isPartOf <https://en.wikipedia.org/> .

      # change P1800 to another property
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

    }
    # remove or change limit for more results
    LIMIT 500
    OFFSET 0"""

    query="""SELECT DISTINCT ?rel
    WHERE
    {
#      ?item wdt:P31 wd:Q11173.
      ?item wdt:P662 ?value.
      ?item ?rel ?obj.

#       ?article schema:about ?item ;
#           schema:isPartOf <https://en.wikipedia.org/> .

      # change P1800 to another property
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

    }
    # remove or change limit for more results
    LIMIT 500
    OFFSET 1"""


    all_rel_id_file = "data_collectors/relid.json"
    """read all """

    args.npages = 10
    results = get_results(endpoint_url, query)
    res = []
    # print(len(results["results"]["bindings"]))
    for result in results["results"]["bindings"]:
        # res.append()
        print(result)


def relid2name(fname="rel_id_constraint_cmpd_enwiki.json"):
    """
    currently forward enwiki constraint, backward free, wikichemprop
    :param fname:
    :return:
    """

    relids = load_file(fname)
    inp = []
    attributes = set()
    predicates = set()

    rels=set()
    for entry in relids:
        rel = entry["rel"]
        if "/prop" in rel or "/entity" in rel:
            rels.add(rel.split("/")[-1])

        # elif "/prop" in rel:
        #     attributes.add(rel.split("/")[-1])
    # print(predicates)
    # print(attributes)
    # print(predicates.difference(attributes))
    # print(attributes.difference(predicates))
    print(len(rels))
    print()
    q = """
        SELECT ?wd ?wddLabel ?Desc WHERE {{
          VALUES ?rel {{  {} }}
          ?wd wikibase:directClaim ?rel. 
          ?wdd wikibase:directClaim ?rel. 
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". 
          ?wd schema:description ?Desc .}}
        }}"""
    q = """
            SELECT ?rel ?lb ?desc ?reltype WHERE {{
              VALUES ?rel {{  {} }}
              ?rel wikibase:propertyType  ?reltype.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". 
              ?rel rdfs:label  ?lb.
              ?rel schema:description ?desc .
              }}
            }}
            ORDER BY DESC(?reltype) """
    res=[]

    tmp_q = q.format(" ".join(["wd:" + rel for rel in rels]) )
    results = get_results(endpoint_url, tmp_q)["results"]["bindings"]
    pp(results)
    print(len(results))
    # exit()
    # if results:
    for result in results:
        uri=result['rel']['value']
        desc=result['desc']['value'] if 'desc' in result else ""
        label=result['lb']['value']
        reltype=result['reltype']['value'].replace("http://wikiba.se/ontology#", "") if 'reltype' in result else ""
        # if " ID" not in label:
        res.append((uri, reltype, label, desc))
        f = pd.DataFrame(res, columns=['uri', 'type','relation/property', 'description', ])
        f.to_csv("rels_constraint_cmpd_enwiki2.csv", index=False)
            # result_val = result['wdLabel']['value']
            # print("result_val", result_val)
            # print("rel", rel)
            # if " ID" not in result_val:
            #     res.append(("https://www.wikidata.org/wiki/Property:" + rel, rel, result_val))
    print("ok")

    # updated_attributes=[]
    # updated_predicates=[]
    # for rel in rels:
    #
    #
    #     tmp_q=q.format("wdt:"+rel)
    #     results = get_results(endpoint_url, tmp_q)["results"]["bindings"]
    #     if results:
    #         if len(results)>1:
    #             print("attribute >1 ", rel)
    #         for result in results:
    #             # result_val=result['wdLabel']['value']
    #             wd = result['wd']['value']
    #             result_val=result['wdLabel']['value']
    #             print("wd",wd)
    #             print("result_val",result_val)
    #             if " ID" not in result_val:
    #                 res.append((wd, result_val))
    #                 f = pd.DataFrame(res, columns=['uri', 'name'])
    #                 f.to_csv("rels_constraint_cmpd_enwiki.csv")
    #                 # dump_file(res, "ewf.pkl")
    #                 # res.append(("https://www.wikidata.org/wiki/Property:"+rel, rel, result_val))
    #     else: print("no result", rel)
    pp(res)
    pp(len(res))
    # print(" ".join(["wdt:" +p for p in list(predicates)]))


relid2name("rel_id_constraint_cmpd_enwiki.json")
# relid2name("ob.json")

# get_rel_id()
