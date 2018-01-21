import jsonify as jsonify
from pytrends.request import TrendReq
import requests
from bs4 import BeautifulSoup

pytrend = TrendReq(hl='en-US', tz=360)

kw_list = ['dress']

pytrend.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')

related_queries_dict = pytrend.related_queries()
print(related_queries_dict[u'dress'][u'rising'].values.T[0])
