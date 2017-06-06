import numpy as np
import chainer.links as L
from chainer import Variable

from .sobamchan_vocabulary import Vocabulary

from gensim.models import KeyedVectors



class PreTrainedEmbedId(L.EmbedID):
    '''
    vocab: instance of sobamchan_vocabulary.Vocabulary
    '''

    def __init__(self, in_size, out_size, vocab, fpath, binary):
        self.fpath = fpath
        self.vocab = vocab

        if binary:
            initialW = self._build_initilW_binary()
        else:
            initialW = self._build_initilW()

        super(PreTrainedEmbedId, self).__init__(
                in_size=in_size,
                out_size=out_size,
                # initialW=initialW
        )
        self.W = Variable(initialW)


    def _build_initilW_binary(self):
        self.word_vectors = KeyedVectors.load_word2vec_format(self.fpath, binary=True)
        vocab = self.vocab
        word_vectors = self.word_vectors
        weight = []
        for wid in sorted(vocab.i2w.keys()):
            word = vocab.i2w[wid]
            # check if word exists in dict
            if word not in word_vectors.index2word:
                word_vector = np.zeros(200)
            else:
                word_vector = word_vectors[word]
            weight.append(word_vector)
        weight = np.array(weight).astype(np.float32)
        self.weight = weight

        return weight

    def _build_initilW(self):
        fpath = self.fpath
        vocab = self.vocab
        word_vectors = {}
        with open(fpath) as f:
            lines = f.readlines()[1:]
        ds = [line.split(' ') for line in lines]
        words = [d[0] for d in ds]
        vecs = [d[1:-1] for d in ds]
        for word, vec in zip(words, vecs):
            word_vectors[word] = [float(i) for i in vec]
        weight = []
        for wid in sorted(vocab.i2w.keys()):
            word = vocab.i2w[wid]
            # check if word exists in dict
            if word not in word_vectors.keys():
                word_vector = np.zeros(300)
            else:
                word_vector = word_vectors[word]
            weight.append(word_vector)
        weight = np.array(weight).astype(np.float32)
        return weight


def test_PreTrainedEmbedId():
    vocab = Vocabulary()
    fpath = '/Users/sochan/project/ML/NLP/datas/word2vec_text8.txt'
    words = ['dog', 'cat', 'cow', 'sheep', 'sobamchan']
    for word in words:
        vocab.new(word)
    ptm = PreTrainedEmbedId(5, 300, vocab, fpath, False)

if __name__ == '__main__':
    test_PreTrainedEmbedId()
