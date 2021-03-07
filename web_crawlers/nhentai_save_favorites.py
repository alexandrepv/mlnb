from bs4 import BeautifulSoup
import os
import requests
import re
import json
import concurrent.futures
import queue
import time

class NHentaiFavoritesDownloader:

    """
    This is a script designed to save all your favorites mangas from nHentai.net.

    Instructions:
    1) Log in into your account to make sure you have a cookie ready for the script
    2) Run "download_mangas()"

    """

    # Constants
    CRAWLER_HEADER = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ' \
                     '(KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    IGNORE_FIELDS = ['uploaded']
    IMAGE_FILENAME_NUM_DIGITS = 4
    MAX_NUM_PAGES = 1000 # Just because I don't like while loops -__-

    def __init__(self, overwrite_downloaded=False, save_tags_in_json=True):

        self.overwrite = overwrite_downloaded
        self.save_tags = save_tags_in_json

        pass

    def download_favorites(self, output_directory: str, session_cookie_content: str) -> None:

        """
        This is a simple implementation of how to download the favorites. You need to be logged in, and
        profide your cookie session content when you
        :param output_directory:
        :param session_cookie_concent:
        :return:
        """

        # Use one thread per manga to speed things up
        manga_list_report = []
        print(f'[ nHentai Favorites Downloader ]')
        for fav_page_number in range(104, self.MAX_NUM_PAGES):

            fav_url = f'https://nhentai.net/favorites/?page={fav_page_number}'

            with requests.session() as s:
                r = requests.get(fav_url, {'user-agent': self.CRAWLER_HEADER},
                                 cookies={'sessionid': session_cookie_content})
                soup = BeautifulSoup(r.content, 'lxml')
                fav_page_contents = soup.find('div', attrs={'class': 'container', 'id': 'favcontainer'})
                if fav_page_contents is None:
                    print(f' > Done')
                    break
                page_manga_list = fav_page_contents.find_all('div', attrs={'class': 'gallery-favorite'})

                for i, manga in enumerate(page_manga_list):
                    nhentai_id = manga.attrs['data-id']
                    print(f' > Manga {len(manga_list_report)+1} ', end='')
                    is_okay = self.download_manga(nhentai_id=nhentai_id,
                                                  output_directory=output_directory,
                                                  overwrite=True)

                    if is_okay:
                        print(f' Successful')
                    else:
                        print(f' Failed')

                    manga_list_report.append(is_okay)
        print(f' > A total of {len(manga_list_report)} mangas were processed.')

    def download_manga(self, nhentai_id: int,
                       output_directory: str,
                       overwrite=False,
                       download_thumbnail=True) -> bool:

        """
        A simple function that download individual mangas into their respecti
        :param nhentai_id: The id (int) of the manga you want to download. It is at the end of the url
        :param output_directory:
        :return:
        """

        url = f'https://nhentai.net/g/{nhentai_id}/'
        with requests.session() as s:
            r = requests.get(url, {'user-agent': self.CRAWLER_HEADER})
            soup = BeautifulSoup(r.content, 'lxml')

            # If "no overwrite", check if there is already a folder and return
            target_output_folder = os.path.join(output_directory, f'nhentai_{nhentai_id}')
            if os.path.exists(target_output_folder) and not overwrite:
                return False

            # Create new folder for the manga
            os.makedirs(target_output_folder, exist_ok=True)

            # ======= Get manga info ========
            manga_info = {}
            info_block = soup.find('div', attrs={'id': 'info-block'})
            if info_block is None:
                raise Exception('[ERROR] No "info_block" found. Please make sure the session is still valid')

            # Title
            title_h1 = info_block.find('h1', attrs={'class': 'title'})
            manga_info['title'] = title_h1.find('span', attrs={'class': 'pretty'}).string

            # Info Block
            info_fields = info_block.find('section', attrs={'id': 'tags'})
            for info_field in info_fields:

                # Get key and data from soup
                field_name_soup = info_field.text.splitlines()[1]
                field_name = re.sub(r'\W+', '', field_name_soup).lower()  # Remove everything that is not alphanumeric
                if field_name in self.IGNORE_FIELDS:
                    continue
                #print(f' > field_name: {field_name}')
                field_values = []
                field_values_soup = info_field.find('span', attrs={'class': 'tags'})
                if field_values_soup is not None:
                    for value_soup in field_values_soup:
                        value = value_soup.find('span', attrs={'class': 'name'}).getText()
                        #print(f'   > {value}')
                        if value is not None:
                            field_values.append(value)

                    # Add to our info dictionary
                    manga_info[field_name] = field_values

            info_fpath = os.path.join(target_output_folder, f'info.json')
            with open(info_fpath, 'w') as file:
                json.dump(manga_info, file, indent=4, sort_keys=True)

            # Images
            gallery_thumbnails_soup_list = soup.find_all('div', attrs={'class': 'thumb-container'})
            for thumb_soup in gallery_thumbnails_soup_list:
                thumbnail_url = thumb_soup.find('img').attrs['data-src']
                re_parser = re.compile(r"https://t.nhentai.net/galleries/(.+d?)/(.+d?)t.(.*)")
                matches = re_parser.search(thumbnail_url)
                scramble_number = int(matches[1])
                image_name = matches[2]
                extension = matches[3]
                image_url = f'https://i.nhentai.net/galleries/{scramble_number}/{image_name}.{extension}'

                # Download images
                number_str = f'{int(image_name)}'.zfill(self.IMAGE_FILENAME_NUM_DIGITS)
                image_fpath = os.path.join(target_output_folder, f'{number_str}.{extension}')
                img_data = requests.get(image_url).content
                with open(image_fpath, 'wb') as file:
                    file.write(img_data)




            num_images = 5




        return True

if __name__ == '__main__':

    app = NHentaiFavoritesDownloader()
    app.download_favorites(output_directory=r'G:\nhentai_manga',
                           session_cookie_content='bt1elgocsjxwwhzfmmi3cz3swmd8ync0')