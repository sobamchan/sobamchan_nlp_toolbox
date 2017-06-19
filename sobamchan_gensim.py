from gensim.models.keyedvectors import KeyedVectors

class Gensim(object):

    def __init__(self):
        pass

    @staticmethod
    def get_word2vec(path):
        if not path:
            print('You need to set the path sir.')
        binary = path.endswith('bin')

        return KeyedVectors.load_word2vec_format(path, binary=binary)
