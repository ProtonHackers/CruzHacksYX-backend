# Link: https://github.com/GeneralMills/pytrends#related-queries
from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)

kw_list = ['Shirt']
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')

print pytrends.related_queries()
