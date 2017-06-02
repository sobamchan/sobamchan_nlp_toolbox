import os
import pickle
import bz2
import tarfile
import csv
import random

import requests
import numpy as np

import xml.etree.ElementTree as ET

class AGCorpus_deplicated:

    def __init__(self, datapath):
        self.tree = ET.parse(datapath)
        self.titles = None
        self.categories = None
        self.descs = None

    def get_data(self):
        titles = [ title_ele.text for title_ele in self.tree.findall('title') ]
        categories = [ category_ele.text for category_ele in self.tree.findall('category') ]
        descs = [ desc_ele.text for desc_ele in self.tree.findall('description') ]

        self.titles = titles
        self.categories = categories
        self.descs = descs

        classes = {'Business': 0, 'Entertainment': 1, 'World': 2, 'Sports': 3}
        X = []
        T = []
        for category, desc in zip(categories, descs):
            if category in classes.keys() and desc is not None:
                X.append(desc)
                T.append(classes[category])

        return T, X




class AgCorpus:

    download_link = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms'

    def __init__(self, save_path):
        self.save_path = save_path
        self.tar_file_path = os.path.join(save_path, 'ag_news_csv.tar.gz')
        self.data_path = os.path.join(save_path, 'ag_news_csv')
        self.train_path = os.path.join(self.data_path, 'train.csv')
        self.test_path = os.path.join(self.data_path, 'test.csv')

    def download(self):
        dest_path = self.tar_file_path
        if os.path.isfile(dest_path):
            return
        r = requests.get(self.download_link, stream=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def expand(self):
        if os.path.isdir(self.data_path):
            return
        f = tarfile.open(self.tar_file_path)
        f.extractall(self.save_path)
        f.close()

    def prepare(self):
        self.download()
        self.expand()

    def load_dataset(self):
        self.prepare()
        with open(self.train_path) as f:
            train_label, train_data = list(zip(*(
                (int(row[0]) - 1, '{} {}'.format(row[1], row[2]).lower())
                for row in csv.reader(f))))
        with open(self.test_path) as f:
            test_label, test_data = list(zip(*(
                (int(row[0]) - 1, '{} {}'.format(row[1], row[2]).lower())
                for row in csv.reader(f))))
        self.train_label = train_label
        self.train_data = train_data
        self.test_label = test_label
        self.test_data = test_data
        return train_label, train_data, test_label, test_data



class MovieReview:

    download_link = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    def __init__(self, save_path):
        self.save_path = save_path
        self.tar_file_path = os.path.join(save_path, 'movie_review.tar.gz')
        self.data_path = os.path.join(save_path, 'aclImdb')
        self.train_path = os.path.join(self.data_path, 'train')
        self.test_path = os.path.join(self.data_path, 'test')

    def download(self):
        dest_path = self.tar_file_path
        if os.path.isfile(dest_path):
            return
        r = requests.get(self.download_link, stream=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def expand(self):
        if os.path.isdir(self.data_path):
            return
        f = tarfile.open(self.tar_file_path)
        f.extractall(self.save_path)
        f.close()

    def prepare(self):
        self.download()
        self.expand()

    def _path_to_data(self, dpath):
        fnames = os.listdir(dpath)
        txt_fnames = [fname for fname in fnames if os.path.isfile(os.path.join(dpath, fname)) and fname.endswith('txt')]
        texts = []
        for txt_fname in txt_fnames:
            fpath = os.path.join(dpath, txt_fname)
            with open(fpath, 'r') as f:
                t = f.read().lower()
            texts.append(t)
        return texts

    def load_dataset(self):
        self.prepare()
        train_pos_path = os.path.join(self.train_path, 'pos')
        train_neg_path = os.path.join(self.train_path, 'neg')
        test_pos_path = os.path.join(self.test_path, 'pos')
        test_neg_path = os.path.join(self.test_path, 'neg')
        train_pos = self._path_to_data(train_pos_path)
        train_neg = self._path_to_data(train_neg_path)
        test_pos = self._path_to_data(test_neg_path)
        test_neg = self._path_to_data(test_neg_path)

        train_data = train_pos + train_neg
        train_label = [1] * len(train_pos) + [0] * len(train_neg)
        test_data = test_pos + test_neg
        test_label = [1] * len(test_pos) + [0] * len(test_neg)

        return train_label, train_data, test_label, test_data
