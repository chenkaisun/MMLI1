from data_collectors.crawler_config import *
import requests


a='https://courses.illinois.edu/search?year=2021&term=spring&keyword=Statistical+Learning&keywordType=qs&partOfTerm=1'
r = requests.get(a,
                 headers=headers)
if r.ok:
    print("ok", r.content)

