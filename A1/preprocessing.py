import numpy as np
import nltk
import os
import io
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation


class preprocessing():
    def __init__(self):
        self.data = None

    def data_preprocess(self):
        path = "./rt-polaritydata/rt-polaritydata/"
        pos_path = os.path.join(path, "rt-polarity.pos")
        neg_path = os.path.join(path, "rt-polarity.neg")
        pos_corpus = io.open(pos_path, encoding='ISO-8859-1').read()
        neg_corpus = io.open(neg_path, encoding='ISO-8859-1').read()

        # remove punctuation and change all to lower case
        pos_corpus = ''.join([i for i in pos_corpus if i not in punctuation])
        neg_corpus = ''.join([i for i in neg_corpus if i not in punctuation])

        # split by new lines and spaces
        pos_corpus = pos_corpus.split('\n')
        neg_corpus = neg_corpus.split('\n')


    #    nltk  https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
    #  lemmatize... https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f


    def get_features(self):
        self.data = None
