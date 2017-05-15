import numpy as np
import chainer.links as L
from chainer import Variable

from sobamchan_vocabulary import Vocabulary

from gensim.models import KeyedVectors



class PreTrainedEmbedId(L.EmbedID):
    '''
    vocab: instance of sobamchan_vocabulary.Vocabulary
    '''

    def __init__(self, in_size, out_size, vocab, binpath):
        self.binpath = binpath
        self.vocab = vocab

        self._load_word_vectors()
        initialW = self._build_initilW()

        super(PreTrainedEmbedId, self).__init__(
                in_size=in_size,
                out_size=out_size,
                initialW=initialW
        )

    def _load_word_vectors(self):
        self.word_vectors = KeyedVectors.load_word2vec_format(self.binpath, binary=True)

    def _build_initilW(self):
        vocab = self.vocab
        word_vectors = self.word_vectors
        weight = []
        for wid in sorted(vocab.i2w.keys()):
            word = vocab.i2w[wid]
            word_vector = word_vectors[word]
            weight.append(word_vector)
        weight = np.array(weight).astype(np.float32)
        self.weight = weight

        return weight

def test_PreTrainedEmbedId():
    vocab = Vocabulary()
    binpath = '~/project/ML/NLP/datas/GoogleNews-vectors-negative300.bin'
    words = ['dog', 'cat', 'cow', 'sheep']
    for word in words:
        vocab.new(word)
    ptm = PreTrainedEmbedId(4, 300, vocab, binpath)
