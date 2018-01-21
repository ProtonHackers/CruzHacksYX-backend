from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.request import urlopen

url = 'http://instagram.com/umnpics/'
o = urlopen(url)
driver = webdriver.Chrome('/Users/rohan/Downloads/chromedriver_win32/chromedriver')
driver.get(url)

soup = BeautifulSoup(o, 'html5lib')

for x in soup.findAll('li', {'class': 'photo'}):
    print(x)

