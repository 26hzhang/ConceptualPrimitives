# -*- coding: utf-8 -*

import os
import re
import time
import json
import platform
import requests
from bs4 import BeautifulSoup


def create_session(url):
    header = {
        'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) Gecko/20100101 Firefox/34.0",
        'Host': "www.youdao.com",
        'Referer': "http://dict.youdao.com/",
        'X-Requested-With': "XMLHttpRequest"
    }
    session = requests.Session()
    r = session.post('http://dict.youdao.com/', headers=header)
    # urllib3.disable_warnings()
    try:
        conte = session.get(url)
    except:
        print("waiting 10 seconds and restart again")
        time.sleep(10)
        create_session(url)
        # conte=session.get(url)
    return conte.content


def lookup_homoionym(word):
    url = 'http://dict.youdao.com/search?q=' + word
    conte = create_session(url)
    soup = BeautifulSoup(conte, 'html.parser')
    hom_words_raw = soup.find_all("div", id='synonyms')[0].find_all("a")
    words = []
    for hom_word in hom_words_raw:
        words.append(hom_word.text.encode('utf-8'))
    return words


a = "like"
print(lookup_homoionym(a))
