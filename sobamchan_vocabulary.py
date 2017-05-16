import collections
from chainer.utils import walker_alias

class Vocabulary(object):

    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.counts = collections.Counter()
        self.dataset = []

    def __len__(self):
        return len(self.w2i)

    def new(self, line):
        '''
        args:
            line (string): string contains words to be added to Vocabulary
        returns:
            None
        '''
        for word in line.strip().split():
            word = word.lower()
            if word not in self.w2i:
                ind = len(self.w2i)
                self.w2i[word] = ind
                self.i2w[ind] = word
            self.counts[self.w2i[word]] += 1
            self.dataset.append(self.w2i[word])

    def encode(self, line):
        w2i = self.w2i
        encoded_line = []
        for word in line.strip().split():
            encoded_line.append(w2i[word])

        return encoded_line
