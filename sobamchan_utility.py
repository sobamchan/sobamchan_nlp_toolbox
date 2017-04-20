import json
import os
import numpy as np

class Utility(object):

    def __init__(self):
        pass

    @staticmethod
    def load_json(filepath):
        with open(filepath, 'r') as f:
            d = json.load(f)
        return d

    @staticmethod
    def save_json(d, filepath):
        with open(filepath, 'w') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    @staticmethod
    def get_n_randomly(arr, n):
        ind = np.random.randint(0, len(arr), n)
        return [arr[i] for i in ind]

    @staticmethod
    def readlines_from_filepath(filepath):
        with open(filepath, 'r') as f:
            d = f.readlines()
        return d
    
    @staticmethod
    def convert_one_of_m_vector_char(sen, token_dictionary, max_len=None, shape=None):
        vec = []
        for w in sen:
            if w in token_dictionary.keys():
                vec.append(token_dictionary[w])
            else:
                vec.append(0)
        if max_len != None and len(vec) < max_len:
            vec = Utility.padding_with_x(vec, max_len-len(vec), 0)
        if max_len != None and len(vec) > max_len:
            vec = vec[:max_len]
        vec = Utility.np_float32(vec)
        if shape:
            vec = vec.reshape(shape)
        return vec

    @staticmethod
    def padding_with_x(arr, n, x):
        arr = arr + [x] * n
        return arr

    @staticmethod
    def separate_datasets(datasets, test_ratio=0.2):
        # TODO
        # pick one dateset shape
        train_ratio = 1.0 - test_ratio
        train = {}
        test = {}
        for key, values in datasets.items():
            train[key] = values[:int(len(values)*train_ratio)]
            test[key] = values[-int(len(values)*test_ratio):]
        return train, test

    @staticmethod
    def np_float32(arr):
        if type(arr) == type(int()) or type(arr) == type(float()):
            arr = [arr]
        return np.array(arr).astype(np.float32)

    @staticmethod
    def np_int32(arr):
        if type(arr) == type(int()) or type(arr) == type(float()):
            arr = [arr]
        return np.array(arr).astype(np.int32)

    @staticmethod
    def count_ext(dir, ext):
        c = 0
        for filename in os.listdir(dir):
            if filename.endswith(ext):
                c += 1
        return c

    @staticmethod
    def separate_array(array, n):
        return [array[x:x + n] for x in range(0, len(array), n)]
