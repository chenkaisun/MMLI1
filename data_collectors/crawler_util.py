
from mediawiki import MediaWiki

wikipedia = MediaWiki(
    user_agent='Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6')

res=wikipedia.prefixsearch('ps')
print(res)
wikipedia.search('washington')
wikipedia.prefixsearch('wefewashington')
p = wikipedia.page('Martha Washington')
print(p.content)

