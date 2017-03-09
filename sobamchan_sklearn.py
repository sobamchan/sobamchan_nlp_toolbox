from sklearn.feature_extraction.text import CountVectorizer

class Sklearn(object):

    def __init__(self):
        pass

    @staticmethod
    def get_count_vectorizer(docs):
        vectorizer = CountVectorizer()
        vectorizer.fit(docs)
        return vectorizer
