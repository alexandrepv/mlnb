import os
import requests
from lxml import etree
import re
import wget
import json
from bs4 import BeautifulSoup

import argparse

if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='Save manga pages from Nhentai.net')
    parser.add_argument('favorites_link',
                        help='an integer for the accumulator')
    parser.add_argument('output_folder',
                        help="A folder will be created with the manga's ID and all pages will be saved in it")

    args = parser.parse_args()

    with requests.session() as s:

        headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
        url = args.favorites_link
        r = requests.get(url, headers)
        soup = BeautifulSoup(r.content, 'html.parser')
        soup.find('input', attrs={'name':'username_or_email', })
        print(r.content)

    r = requests.get()
    assert r.status_code == 200, f"[ERROR] HTTP request was not successful! Return code {r.status_code}"
    html = etree.HTML(r.text)

    section_pages_list = html.xpath('//section/a/@href')

    # Create dictionary with details of the manga
    all_links_list = html.xpath('//a[@href]')
    thumbnails_links = [elem.attrib['href'] for elem in all_links_list
                        if 'class' in elem.attrib
                        if elem.attrib['class'] == 'gallerythumb']

    f = 0