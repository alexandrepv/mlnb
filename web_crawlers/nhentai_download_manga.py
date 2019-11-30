import os
import requests
from lxml import etree
import re
import wget
import json

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save manga pages from Nhentai.net')
    parser.add_argument('manga_link',
                        help='an integer for the accumulator')
    parser.add_argument('output_folder',
                        help="A folder will be created with the manga's ID and all pages will be saved in it")

    args = parser.parse_args()

    # Constants
    website = 'https://nhentai.net'
    filename_num_digits = 3

    r = requests.get(args.manga_link)
    assert r.status_code == 200, f"[ERROR] HTTP request was not successful! Return code {r.status_code}"
    html = etree.HTML(r.text)

    # Create new folder for manga pages to be downloaded
    manga_id = re.search(r'/g/([0-9]*)/', args.manga_link).group(1)
    manga_folder = os.path.join(args.output_folder, manga_id)
    os.makedirs(manga_folder, exist_ok=True)

    # Create dictionary with details of the manga
    info_list = html.xpath('//span/a/@href')
    info_dict = {'artist': [],'tag': [],'language': [],'category': []}

    for item in info_list:
        matches = re.search(r'/(.*)/(.*)/', item)
        if matches.group(1) in info_dict:
            info_dict[matches.group(1)].append(matches.group(2))

    info_fpath = os.path.join(manga_folder, 'info.json')
    with open(info_fpath, 'w') as fp:
        json.dump(info_dict, fp)


    # Get link for all pages to be requested
    all_links_list = html.xpath('//a[@href]')
    thumbnails_links = [elem.attrib['href'] for elem in all_links_list
                        if 'class' in elem.attrib
                        if elem.attrib['class'] == 'gallerythumb']

    # Request each page an save the full resolution image into the new folder
    for i, link in enumerate(thumbnails_links):

        print(f'Saving picture {i+1}/{len(thumbnails_links)}... ', end='')

        # Get page from thumbnail link
        complete_link = website + link
        r = requests.get(complete_link)
        html = etree.HTML(r.text)

        full_res_image_etree = html.xpath("//section[@id='image-container']/a//img")
        download_link = full_res_image_etree[0].attrib['src']
        file_extension = os.path.splitext(download_link)[1][1:]
        filename = f'{i+1}'.rjust(filename_num_digits, '0')+f'.{file_extension}'
        image_fpath = os.path.join(manga_folder, filename)
        if os.path.isfile(image_fpath):
            print('already exists.')
        else:
            wget.download(download_link, image_fpath)
            print('done.')

