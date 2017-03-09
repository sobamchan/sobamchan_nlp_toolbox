from lxml import html
import requests
from bs4 import BeautifulSoup

class Crawler(object):

    def __init__(self):
        pass

    @staticmethod
    def get_lxml_tree_url(url, auth=None):
        r = requests.get(url, auth=auth)
        tree = html.fromstring(r.text)
        return tree

    @staticmethod
    def get_soup(url, auth=None):
        r = requests.get(url, auth=auth)
        return BeautifulSoup(r.text.encode(r.encoding), 'lxml')

    @staticmethod
    def download(url, output_dir, auth=None):
        file_name = url.split("/")[-1]
        output_path = '{}/{}'.format(output_dir, file_name)
        res = requests.get(url, stream=True, auth=auth)
        if res.status_code == 200:
            with open(output_path, 'wb') as file:
                for chunk in res.iter_content(chunk_size=1024):
                    file.write(chunk)
