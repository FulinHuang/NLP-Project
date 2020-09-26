import numpy as np
import nltk
import os
import io
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from string import punctuation


class preprocessing():
    def __init__(self):
        self.corpus = None
        self.labels = None

    def data_preprocess(self):
        path = "./rt-polaritydata/rt-polaritydata/"
        pos_path = os.path.join(path, "rt-polarity.pos")
        neg_path = os.path.join(path, "rt-polarity.neg")
        pos_corpus = io.open(pos_path, encoding='ISO-8859-1').read()
        neg_corpus = io.open(neg_path, encoding='ISO-8859-1').read()

        # remove punctuation and change all to lower case  ****** TODO: remove this?
        pos_corpus = ''.join([i for i in pos_corpus if i not in punctuation])
        neg_corpus = ''.join([i for i in neg_corpus if i not in punctuation])

        # split by new lines and spaces
        pos_corpus = pos_corpus.split('\n')
        neg_corpus = neg_corpus.split('\n')

        # create labels
        pos_labels = np.repeat(1, len(pos_corpus))
        neg_labels = np.repeat(0, len(neg_corpus))

        # combine pos and neg corpus
        self.corpus = pos_corpus + neg_corpus
        self.labels = pos_labels + neg_labels

        return self.corpus, self.labels


    def get_features(self, sentence, type):
        # tokenize the sentence
        words = word_tokenize(sentence)

        if type == "stopword":
            stop_words = set(stopwords.words("english"))
            output = [" ".join(i for i in words if not i in stop_words)]

        elif type == "lemmatize":
            lemmatizer = WordNetLemmatizer()
            output = [" ".join(lemmatizer.lemmatize(i) for i in words)]

        elif type == "stem":
            stemmer = PorterStemmer()
            output = [" ".join(stemmer.stem(i) for i in words)]
        else:
            raise Exception('Please type the correct type!')

        return output

#    nltk  https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
    #  lemmatize... https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f

