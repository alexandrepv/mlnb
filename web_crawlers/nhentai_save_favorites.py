from bs4 import BeautifulSoup
import os
import requests
import re
import json
import concurrent.futures
import queue
import time
import pandas as pd
from viztracer import VizTracer



"""
This is a script designed to save all your favorites mangas from nHentai.net.

Instructions:
1) Log in into your account to make sure you have a cookie ready for the script
2) Run "download_mangas()"

"""

MAX_PROCESSES = 6

RESPONSE_FAILED = 0
RESPONSE_EXISTS = 1
RESPONSE_SUCCESSFUL = 2

# Constants
CRAWLER_HEADER = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ' \
                 '(KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
IGNORE_FIELDS = ['uploaded']
IMAGE_FILENAME_NUM_DIGITS = 4
MAX_NUM_FAV_PAGES = 120  # Just because I don't like while loops -__-
MAX_NUM_REQUEST_ATTEMPTS = 5
WAIT_TIME_BETTEEN_ATTEMPTS = 0.5


def download_manga(nhentai_id: int,
                   output_directory: str) -> (int, str, float):

    """
    A simple function that download individual mangas into their respecti
    :param nhentai_id: The id (int) of the manga you want to download. It is at the end of the url
    :param output_directory:
    :return:
    """
    result = {}

    elapsed_time = -1
    t0 = time.perf_counter()
    url = f'https://nhentai.net/g/{nhentai_id}/'
    with requests.session() as s:

        # Request the page, and if we get "too many attempts" try a few more times
        for i in range(MAX_NUM_REQUEST_ATTEMPTS):
            r = requests.get(url, {'user-agent': CRAWLER_HEADER})
            if r.status_code == 200:
                break
            else:
                print(f" > Server returned '{r.status_code}'. Trying again.")
                time.sleep(WAIT_TIME_BETTEEN_ATTEMPTS)

        soup = BeautifulSoup(r.content, 'lxml')

        # Check if manga was downloaded successfully
        target_output_folder = os.path.join(output_directory, f'nhentai_{nhentai_id}')
        if os.path.exists(target_output_folder) and manga_is_complete(target_output_folder):
            result['nhentai_id'] = nhentai_id
            result['code'] = RESPONSE_EXISTS
            result['description'] = 'Already downloaded. '
            result['elapsed_time'] = elapsed_time
            return result

        # Create new folder for the manga
        os.makedirs(target_output_folder, exist_ok=True)

        # ======= Get manga info ========
        manga_info = {}
        info_block = soup.find('div', attrs={'id': 'info-block'})
        if info_block is None:
            elapsed_time = time.perf_counter() - t0
            result['nhentai_id'] = nhentai_id
            result['code'] = RESPONSE_FAILED
            result['description'] = "Could not locate 'info_block'"
            result['elapsed_time'] = elapsed_time
            return result

        # Title
        title_h1 = info_block.find('h1', attrs={'class': 'title'})
        manga_info['title'] = title_h1.find('span', attrs={'class': 'pretty'}).string

        # Info Block
        info_fields = info_block.find('section', attrs={'id': 'tags'})
        if info_fields is None:
            elapsed_time = time.perf_counter() - t0
            result['nhentai_id'] = nhentai_id
            result['code'] = RESPONSE_FAILED
            result['description'] = "Could not locate 'info_fields'"
            result['elapsed_time'] = elapsed_time
            return result

        for info_field in info_fields:

            # Get key and data from soup
            field_name_soup = info_field.text.splitlines()[1]
            field_name = re.sub(r'\W+', '', field_name_soup).lower()  # Remove everything that is not alphanumeric
            if field_name in IGNORE_FIELDS:
                continue
            field_values = []
            field_values_soup = info_field.find('span', attrs={'class': 'tags'})
            if field_values_soup is not None:
                for value_soup in field_values_soup:
                    value = value_soup.find('span', attrs={'class': 'name'}).getText()
                    if value is not None:
                        field_values.append(value)

                # Add to our info dictionary
                manga_info[field_name] = field_values

        info_fpath = os.path.join(target_output_folder, f'info.json')
        with open(info_fpath, 'w') as file:
            json.dump(manga_info, file, indent=4, sort_keys=True)

        # Download Images
        gallery_thumbnails_soup_list = soup.find_all('div', attrs={'class': 'thumb-container'})
        for thumb_soup in gallery_thumbnails_soup_list:
            thumbnail_url = thumb_soup.find('img').attrs['data-src']
            re_parser = re.compile(r"https://t.nhentai.net/galleries/(.+d?)/(.+d?)t.(.*)")
            matches = re_parser.search(thumbnail_url)
            scramble_number = int(matches[1])
            image_name = matches[2]
            extension = matches[3]
            image_url = f'https://i.nhentai.net/galleries/{scramble_number}/{image_name}.{extension}'
            number_str = f'{int(image_name)}'.zfill(IMAGE_FILENAME_NUM_DIGITS)
            image_fpath = os.path.join(target_output_folder, f'{number_str}.{extension}')
            img_data = requests.get(image_url).content
            with open(image_fpath, 'wb') as file:
                file.write(img_data)

    elapsed_time = time.perf_counter() - t0
    result['nhentai_id'] = nhentai_id
    result['code'] = RESPONSE_SUCCESSFUL
    result['description'] = " Downloaded successfully"
    result['elapsed_time'] = elapsed_time
    return result

def manga_is_complete(manga_directory: str):

    """
    Checks the contents of the folder to see if all pages have been downloaded. 
    It reads the "info.json"
    :param mang_fpath: 
    :return: True if
    """
    json_fpath = os.path.join(manga_directory, 'info.json')

    if not os.path.exists(json_fpath):
        return False

    with open(json_fpath) as file:
        info = json.load(file)
        num_pages = int(info['pages'][0])
        files_in_dir = os.listdir(manga_directory)
        return True if len(files_in_dir) == (num_pages + 1) else False


def download_favorites(output_directory: str, session_cookie_content: str) -> None:

    """
    This is a simple implementation of how to download the favorites. You need to be logged in, and
    profide your cookie session content when you
    :param output_directory:
    :param session_cookie_concent:
    :return:
    """

    results_list = []

    #tracer = VizTracer()
    #tracer.start()

    time_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROCESSES) as executor:

        # Use one thread per manga to speed things up
        print(f'[ nHentai Favorites Downloader : {MAX_PROCESSES} workers]')
        for fav_page_number in range(1, MAX_NUM_FAV_PAGES):

            fav_url = f'https://nhentai.net/favorites/?page={fav_page_number}'

            with requests.session() as s:
                time.sleep(1)
                r = requests.get(fav_url, {'user-agent': CRAWLER_HEADER},
                                 cookies={'sessionid': session_cookie_content})
                soup = BeautifulSoup(r.content, 'lxml')
                fav_page_contents = soup.find('div', attrs={'class': 'container', 'id': 'favcontainer'})
                if fav_page_contents is None:
                    print(f' > Done')
                    break
                page_manga_list = fav_page_contents.find_all('div', attrs={'class': 'gallery-favorite'})

                print(f' > Starting favorites page {fav_page_number}')
                fav_page_results = {executor.submit(download_manga,
                                                    manga.attrs['data-id'],
                                                    output_directory): manga for manga in page_manga_list}
                for future in concurrent.futures.as_completed(fav_page_results):
                    result = future.result()
                    results_list.append(result)
                    print(f" > {result['nhentai_id']} {result['description']} ({result['elapsed_time']:.1f} secs)")

    time_stop = time.time()

    # Analyse results
    successful_list = [manga for manga in results_list if manga['code'] == RESPONSE_SUCCESSFUL]
    already_download_list = [manga for manga in results_list if manga['code'] == RESPONSE_EXISTS]
    failed_list = [manga for manga in results_list if manga['code'] == RESPONSE_FAILED]

    print(f' > Total Elapsed time: {time_stop-time_start:.1} seconds')
    print(f' > {len(successful_list)} New mangas downloaded')
    print(f' > {len(already_download_list)} Mangas already existed')
    print(f' > {len(failed_list)} Failed during download\n')

    print('[ Failed List ]')
    for manga in failed_list:
        print(f" > {manga['nhentai_id']}")

    #tracer.stop()
    #tracer.save()



if __name__ == '__main__':

    download_favorites(output_directory=r'G:\nhentai_manga',
                       session_cookie_content='bt1elgocsjxwwhzfmmi3cz3swmd8ync0')
