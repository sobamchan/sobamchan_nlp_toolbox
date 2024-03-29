import collections
import pickle
import json


class Vocabulary(object):

    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.counts = collections.Counter()
        self.dataset = []
        self.replaced = False

    def __len__(self):
        return len(self.w2i)

    def new(self, line):
        '''
        args:
            line (string): string contains words to be added to Vocabulary
        returns:
            None
        '''
        self.w2i['<UNK>'] = '0'
        self.i2w['0'] = '<UNK>'
        for word in line.strip().split():
            word = word.lower()
            if word not in self.w2i:
                ind = str(len(self.w2i))
                self.w2i[word] = ind
                self.i2w[ind] = word
            self.counts[self.w2i[word]] += 1
            self.dataset.append(self.w2i[word])

    def replace_rare_unk(self, min_occ_n):
        new_counter = collections.Counter()
        new_w2i = {}
        new_i2w = {}
        # add <UNK>
        new_w2i['<UNK>'] = '0'
        new_i2w['0'] = '<UNK>'
        for k, v in self.counts.items():
            if v > min_occ_n:
                word = self.i2w[k]
                new_ind = str(len(new_w2i))
                new_w2i[word] = new_ind
                new_i2w[new_ind] = word
                new_counter[new_ind] = self.counts[k]
            else:
                new_counter['0'] += self.counts[k]
        self.counts = new_counter
        self.i2w = new_i2w
        self.w2i = new_w2i
        self.replaced = True

    def encode(self, line):
        w2i = self.w2i
        encoded_line = []
        for word in line.strip().split():
            word = word.lower()
            if word in self.w2i.keys():
                encoded_line.append(w2i[word])
            else:
                encoded_line.append(w2i['<UNK>'])

        return encoded_line

    def save(self, fpath):
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    def to_json(self, fpath):
        d = {}
        d['i2w'] = self.i2w
        d['w2i'] = self.w2i
        d['counts'] = self.counts
        d['replaced'] = self.replaced
        d['dataset'] = self.dataset
        with open(fpath, 'w') as f:
            json.dump(d, f)

    @staticmethod
    def load_json(fpath):
        with open(fpath, 'r') as f:
            d = json.load(f)
        v = Vocabulary()
        v.i2w = d['i2w']
        v.w2i = d['w2i']
        v.counts = collections.Counter(d['counts'])
        v.replaced = d['replaced']
        v.dataset = d['dataset']
        return v

    @staticmethod
    def load(fpath):
        with open(fpath, 'rb') as f:
            vocab = pickle.load(f)
        return vocab
