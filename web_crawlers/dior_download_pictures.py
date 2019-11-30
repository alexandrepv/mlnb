import os
import requests
from lxml import etree
import re
import wget


import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save manga pages from Nhentai.net')
    parser.add_argument('output_folder',
                        help="A folder will be created with the manga's ID and all pages will be saved in it")

    args = parser.parse_args()

    # Constants
    root_page = 'https://www.dior.com/en_gb/womens-fashion/'
    targets = [('ready-to-wear/all-ready-to-wear', 'ready_to_wear'),
               ('shoes/all-shoes', 'shoes'),
               ('fashion-jewellery/jewellery', 'jewellery'),
               ('accessories/all-accessories', 'accessories')]

    targets = [('accessories/all-accessories', 'accessories')]

    for target in targets:

        # Get HTML page
        r = requests.get(root_page + target[0])
        assert r.status_code == 200, f"[ERROR] HTTP request was not successful! Return code {r.status_code}"
        html = etree.HTML(r.text)

        # Create folder for specifit target
        download_folder = os.path.join(args.output_folder, target[1])
        os.makedirs(download_folder, exist_ok=True)

        # Download all images from that page into newly created folder
        product_list = html.xpath("//a[@class='product-link']")
        image_list = html.xpath("//img[@src]")
        for image_elem in image_list:

            matches = re.search(r'https://media.dior.com/img/en_gb/sku/couture/(.*)', image_elem.attrib['src'])

            if not matches:
                continue

            image_name = f'{matches.group(1)}.jpg'
            image_name = image_name.replace('/','_')
            image_name = image_name.replace('\\', '_')

            image_fpath = os.path.join(download_folder, image_name)

            if os.path.isfile(image_fpath):
                print(f'[{target[1]}] {image_fpath} already exists.')
            else:
                wget.download(image_elem.attrib['src'], image_fpath)
                print(f'[{target[1]}] {image_fpath} downloaded.')
