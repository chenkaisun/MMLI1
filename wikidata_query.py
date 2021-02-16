# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/

from SPARQLWrapper import SPARQLWrapper, JSON
import requests

from utils import *
import argparse
import sys
parser = argparse.ArgumentParser()

parser.add_argument('--npages', default=1000000, type=int)

args = parser.parse_args()

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
mkdir("data_online/")
html_folder="data_online/wikipedia_html_for_pubchem_compounds/"
mkdir(html_folder)
# query all wikidata pubchem cid entries with a wikipedia link
# query batches of 1000
for i in range(0,args.npages,1000):
    print("i",i)


    results = get_results(endpoint_url, query%i)


    for result in results["results"]["bindings"]:
        # print(result)
        res[int(result['value']['value'])]=result
        try:
            r = requests.get(result['article']['value'], headers=headers)
            if r.ok:
                html = r.text
                dump_file(html, html_folder+result['value']['value']+".html")
        except Exception as e:
            print(e)
    dump_file(res, "data/pubchemcid2wikiitem.json")



