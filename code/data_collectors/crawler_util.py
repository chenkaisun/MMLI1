import requests
from mediawiki import MediaWiki
# from data_collectors.crawler_config import *
from bs4 import BeautifulSoup
from IPython import embed
import json
from data_collectors.crawler_config import headers
from utils import load_file, dump_file, join
import urllib.request

def download(urls, save_dir):
    for url in urls:
        with urllib.request.urlopen(url) as f:
            item = f.read().decode('utf-8')
            dump_file(item, join(save_dir, url.split("/")[-1]))


def request_get(url, headers):
    try:
        r = requests.get(url, headers=headers)
        if r.ok:
            # print(r)
            return r
        else:

            print(r)
            return None
    except Exception as e:
        print(e)
        return None

def te_fn(a):
    print(a)


def search_wiki():

    wikipedia = MediaWiki(
        user_agent='Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6')

    res=wikipedia.prefixsearch('ps')
    print(res)
    wikipedia.search('washington')
    wikipedia.prefixsearch('wefewashington')
    p = wikipedia.page('Martha Washington')
    print(p.content)

def get_mole_desciption(r):
    result = r.content
    soup = BeautifulSoup(result, 'lxml')
    sinple_item={}
    for information in soup.find_all("information"):
        CID = int(information.find('cid').get_text())
        sinple_item["cid"]=CID
        if information.find("title"):
            sinple_item["title"]=information.find("title").get_text()

        if information.find("description"):
            if "descriptions" not in sinple_item:
                sinple_item["descriptions"] = []
            description=information.find("description").get_text()
            descriptionsourcename=information.find("descriptionsourcename").get_text()
            descriptionurl=information.find("descriptionurl").get_text()
            sinple_item["descriptions"].append({"description":description,
                                                "descriptionsourcename":descriptionsourcename,
                                                "descriptionurl":descriptionurl,})
    return sinple_item



# r = request_get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{1983}/description/XML', headers)
# get_mole_desciption(r)


def get_concepts(r):
    result = r.content
    concept_dict=json.loads(result)
    return concept_dict
    # soup = BeautifulSoup(result, 'lxml')
    # print(soup)
    # embed()


    # sinple_item={}
    # for information in soup.find_all("KeyValueOfstringdouble"):
    #     CID = int(information.find('cid').get_text())
    #     sinple_item["cid"]=CID
    #     if information.find("title"):
    #         sinple_item["title"]=information.find("title").get_text()
    #
    #     if information.find("description"):
    #         if "descriptions" not in sinple_item:
    #             sinple_item["descriptions"] = []
    #         description=information.find("description").get_text()
    #         descriptionsourcename=information.find("descriptionsourcename").get_text()
    #         descriptionurl=information.find("descriptionurl").get_text()
    #         sinple_item["descriptions"].append({"description":description,
    #                                             "descriptionsourcename":descriptionsourcename,
    #                                             "descriptionurl":descriptionurl,})
    # return sinple_item




def request(url, headers):
    try:
        r = requests.get(url, headers=headers)
        if r.ok:
            # print(r)
            return r
        else:
            print(r)
            return None
    except Exception as e:
        print(e)
        return None