# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/

import sys
import time
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from bs4 import BeautifulSoup
from utils import *

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/39.0.2171.95 Safari/537.36'}

endpoint_url = "https://query.wikidata.org/sparql"

query = """#All items with a property
# Sample to query all values of a property
# Property talk pages on Wikidata include basic queries adapted to each property
SELECT
  ?item ?itemLabel
  ?value
  ?article
# valueLabel is only useful for properties with item-datatype
WHERE 
{
  ?item wdt:P662 ?value.
  ?article schema:about ?item ;
      schema:isPartOf <https://en.wikipedia.org/> .

  # change P1800 to another property        
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
# remove or change limit for more results
LIMIT 1000
OFFSET %d"""


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

res={}
html_folder="wikipedia_html_for_pubchem_compounds/"
mkdir(html_folder)
# query all wikidata pubchem cid entries with a wikipedia link
# query batches of 1000
for i in range(0,1000000,1000):


    results = get_results(endpoint_url, query%i)


    for result in results["results"]["bindings"]:
        # print(result)
        res[int(result['value']['value'])]=result
        try:
            r = requests.get(result['article']['value'], headers=headers)
            if r.ok:
                html = r.text
                dump_file(html, html_folder+result['value']['value']+".html")
        except:
            pass
    dump_file(res, "pubchemcid2wikiitem.json")



