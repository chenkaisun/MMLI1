import requests
from bs4 import BeautifulSoup

from utils import *
# api
from mediawiki import MediaWiki

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/39.0.2171.95 Safari/537.36'}
wikipedia = MediaWiki(
    user_agent='Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6')
# http_proxy = "http://64.64.242.115:50000"
# https_proxy="http://6f8852f5-f73e-48cb-92e2-680f14ea1798:@host"
# # https_proxy = "https://64.64.242.115:50000"
# ftp_proxy = "ftp://64.64.242.115:50000"
# r = requests.get(f'https://en.wikipedia.org/wiki/Titanium', headers=headers,
#                  proxies={"https": https_proxy})

soup = BeautifulSoup(open("tmp.html", "r", encoding='utf-8'), 'html.parser')
title = soup.find('title').get_text()
summary = soup.find('div', attrs={"class": 'summary p-md-top p-md-bottom'})


mkdir("wikipedia_html_for_pubchem_compounds")
r = requests.get(f'https://en.wikipedia.org/wiki/Coeloginin', headers=headers)
if r.ok:
    html = r.text
    dump_file(html, "tmp.html")
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('title').get_text()
    dump_file(html, "tmp.html")
