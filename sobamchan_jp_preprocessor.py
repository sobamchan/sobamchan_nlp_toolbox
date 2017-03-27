'''
文章を一つ受け取って名詞のみ'str str str'で返したい
'''
import MeCab
from jp_preprocess_lib.normalize_neologd import normalize_neologd
from jp_preprocess_lib.detagger import TagStripper

class Parser:

    def __init__(self, pos=None, stemming=False, normalize=True, html_detag=False, ngram=None):
        self.tagger = MeCab.Tagger('-Ochasen')
        self.tagger.parse('')
        self.pos = pos
        self.stemming = stemming
        self.normalize = normalize
        self.html_detag = html_detag
        self.ngram = ngram

    def __check_intput_type(self, sentences):
        return type(sentences) == type(str())

    def __wrap_with_list(self, sentences):
        return [sentences]

    def html_detagger(self, sentences):
        tag_stripper = TagStripper()
        detagged_sentences = [ tag_stripper.strip(s) for s in sentences ]
        return detagged_sentences

    def normalizer(self, sentences):
        normalized_sentences = [ normalize_neologd(s) for s in sentences ]
        return normalized_sentences

    def wakati(self, sentence):
        node = self.tagger.parseToNode(sentence)
        list_owakati = []
        while node:
            features = node.feature.split(',')
            if self.stemming and features[-3] != '*':
                word = features[-3]
            else:
                word = node.surface
            if features[0] == 'BOS/EOS':
                pass
            else:
                if self.pos == None:
                    list_owakati.append(word)
                elif features[0] in self.pos:
                    list_owakati.append(word)
            node = node.next

        return ' '.join(list_owakati)

    def ngramer(self, wakatied_sentence):
        if self.ngram < 0:
            return wakatied_sentence
        wakatied_sentence_li = wakatied_sentence.strip().split()
        tokens = []
        for i in range(len(wakatied_sentence_li)):
            if len(wakatied_sentence_li[i:i+self.ngram]) >= self.ngram:
                tokens.append(wakatied_sentence_li[i:i+self.ngram])

        return tokens

    def __call__(self, sentences):
        if self.__check_intput_type(sentences):
            sentences = self.__wrap_with_list(sentences)
        if self.html_detag:
            sentences = self.html_detagger(sentences)
        if self.normalize:
            sentences = self.normalizer(sentences)
        wakatied_sentences = [ self.wakati(s) for s in sentences ]
        if self.ngram:
            wakatied_sentences = [ self.ngramer(s) for s in wakatied_sentences ]
        return wakatied_sentences


if __name__ == '__main__':
    parser = Parser(['名詞'])
    print(parser('私　はサッカーボールを蹴っています。'))
