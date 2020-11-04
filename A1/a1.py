"""
COMP 550 Natural Language Processing
Assignment 1

Fulin Huang
260740689

"""

import argparse
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sn
import os
import io
from collections import defaultdict
from string import punctuation

import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


def data_preprocess():
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

    # create labels
    pos_labels = np.repeat(1, len(pos_corpus))
    neg_labels = np.repeat(0, len(neg_corpus))

    # combine pos and neg corpus
    corpus = pos_corpus + neg_corpus
    labels = np.concatenate((pos_labels, neg_labels))

    return corpus, labels

def get_features(sentence, type):
    # tokenize the sentence
    words = word_tokenize(sentence)

    if type == "stopword":
        stop_words = set(stopwords.words("english"))
        output = [" ".join(i for i in words if not i in stop_words)]  # convert back to a sentence

    elif type == "lemmatize":
        tag_map = defaultdict(lambda: wordnet.NOUN)
        tag_map['J'] = wordnet.ADJ
        tag_map['V'] = wordnet.VERB
        tag_map['R'] = wordnet.ADV

        lemmatizer = WordNetLemmatizer()
        output = [" ".join(lemmatizer.lemmatize(i, tag_map[pos_tag(i)[0]]) for i in words)]

    elif type == "stem":
        stemmer = PorterStemmer()
        output = [" ".join(stemmer.stem(i) for i in words)]
    else:
        raise Exception('Please type the correct type!')

    return output


def split_data(corpus, label):
    num_instances = len(label)
    n_valid, n_test = math.floor(num_instances * 0.2), math.floor(num_instances * 0.1)
    index = np.random.permutation(num_instances)
    X_valid, y_valid = corpus[index[:n_valid]], label[index[:n_valid]]
    X_test, y_test = corpus[index[n_valid:n_valid+n_test]], label[index[n_valid:n_valid+n_test]]
    X_train, y_train = corpus[index[n_valid+n_test:]], label[index[n_valid+n_test:]]
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def lr_classifier(X_train, y_train, X_test, y_test, type):

    if(type == 'validation'):
        # **** Use GridSearchCV during Test time. The best one is when C = 1 **** #
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
        print(clf)
        best_clf = clf.fit(X_train, y_train)
        y_predict = best_clf.predict(X_test)
        print(clf.best_params_)
        print("Accuracy is ", accuracy_score(y_test, y_predict))
        confusion_matrix(y_test, y_predict)

    elif(type == 'test'):
        clf = LogisticRegression(C = 1).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        print("Accuracy is ", accuracy_score(y_test, y_predict))
        confusion_matrix(y_test, y_predict)

    else:
        raise Exception('Please type the correct type!')


def svm_classifier(X_train, y_train, X_test, y_test, type):
    if(type == 'validation'):
        param_grid = {'C':[1, 10, 100, 1000], 'gamma':[0.001, 0.01, 0.1, 1]}
        print("start gridsearch")
        clf = GridSearchCV(svm.SVC(kernel='linear'), param_grid)
        best_clf = clf.fit(X_train, y_train)
        y_predict = best_clf.predict(X_test)
        print(clf.best_params_)

        print("Accuracy is ", accuracy_score(y_test, y_predict))

        confusion_matrix(y_test, y_predict)

    elif(type == 'test'):
        clf = svm.SVC(kernel='rbf', C = 1)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)

        print("Accuracy is ", accuracy_score(y_test, y_predict))
        confusion_matrix(y_test, y_predict)

    else:
        raise Exception('Please type the correct type!')


def nb_classifier(X_train, y_train, X_test, y_test, type):
    if(type == 'validation'):
        param_grid = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001]}
        clf = GridSearchCV(MultinomialNB(), param_grid)
        best_clf = clf.fit(X_train, y_train)
        y_predict = best_clf.predict(X_test)
        print(clf.best_params_)
        print("Accuracy is ", accuracy_score(y_test, y_predict))
        confusion_matrix(y_test, y_predict)

    elif(type == 'test'):
        gnb = MultinomialNB(alpha = 1).fit(X_train, y_train)
        y_predict = gnb.predict(X_test)
        print("Accuracy is ", accuracy_score(y_test, y_predict))

        confusion_matrix(y_test, y_predict)

    else:
        raise Exception('Please type the correct type!')


def random_baseline(y_test):
    num_instances = len(y_test)
    y_predict = np.random.randint(0, 1, num_instances)
    print("Random baseline accuracy is ", accuracy_score(y_test, y_predict))

def confusion_matrix(y_test, y_predict):
    data = {'y_Actual': y_test,'y_Predicted': y_predict}

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)
    sn.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.show()


def main(args):

    if args.validation:
        corpus, label = data_preprocess()

        # *** Uncomment one/two of the next three lines to test different feature extraction methods ***
        corpus = [" ".join(get_features(sentence, "stopword")) for sentence in corpus] # if include stopword
        corpus = [" ".join(get_features(sentence, "stem")) for sentence in corpus]
        # corpus = [" ".join(get_features(sentence, "lemmatize")) for sentence in corpus]

        # convert text to vectors
        vectorizer = CountVectorizer(corpus, ngram_range=(1,1))
        new_corpus = vectorizer.fit_transform(corpus).toarray()

        X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(new_corpus, label)

        # *** Uncomment one of the three lines to test different models ***
        # lr_classifier(X_train, y_train, X_valid, y_valid, 'validation')
        # svm_classifier(X_train, y_train, X_valid, y_valid, 'validation')
        nb_classifier(X_train, y_train, X_valid, y_valid, 'validation')

        random_baseline(y_valid)

    if args.test:
        # This will give use the result
        corpus, label = data_preprocess()
        new_corpus = [" ".join(get_features(sentence, "stem")) for sentence in corpus]
        # convert text to vectors
        vectorizer = CountVectorizer(new_corpus, ngram_range=(1, 1))
        new_corpus = vectorizer.fit_transform(new_corpus).toarray()

        X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(new_corpus, label)
        nb_classifier(X_train, y_train, X_test, y_test, 'test')

        random_baseline(y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run data preprocessing and perform classification')

    parser.add_argument('--validation', dest='validation', action='store_true', default=False, help='Do experiment on the validation set using four different models')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='Perform classification on the test set using naive bayes model')


    args = parser.parse_args()
    main(args)