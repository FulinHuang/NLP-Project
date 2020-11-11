import loader
import nltk
import collections
import numpy as np
nltk.download('wordnet')
nltk.download('stopwords')
from collections import defaultdict
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


import xml.etree.cElementTree as ET
import codecs

def data_preprocess(instance):
    for (k, v) in instance.items():
        stop_words = set(stopwords.words("english"))
        stop_words.add('\'s')
        stop_words.add('\'\'')
        stop_words.add('--')
        stop_words.add('``')
        stop_words.add('@card@')

        # remove stop word & punctuation & lower case
        v.context = [i.decode("utf-8") for i in v.context]
        v.context = [i for i in v.context if i not in stop_words]
        v.context = [i for i in v.context if i not in punctuation]
        v.context = [i.lower() for i in v.context]

        # lemmatize
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        lemmatizer = WordNetLemmatizer()
        v.context = [lemmatizer.lemmatize(words, tag_map[pos_tag(words)[0]]) for words in v.context]
        v.context = [i.encode("utf-8") for i in v.context]

    return instance

#  The most frequent sense baseline: this is the sense indicated as #1 in the synset according to WordNet
def baseline(instance):
    dict = {k: wn.synsets(v.lemma.decode("utf-8"))[0] for k, v in instance.items()}
    return dict


def lesk_wsd(instance):
    dict = {k: nltk.wsd.lesk(v.context, v.lemma.decode("utf-8")) for k, v in instance.items()}
    return dict

def find_context(lexical_item, instance, key):
    print("key", key)
    context_dict = defaultdict(list)
    synset_dict = defaultdict(list)
    for item in lexical_item:
        context_sublist = []
        synset_sublist = []
        for (k, v) in instance.items():
            if v.lemma.decode("utf-8") == item:
                context = [i.decode("utf-8") for i in v.context]
                context = ' '.join(context)
                context_sublist.append(context)
                synset = [wn.lemma_from_key(i).synset() for i in key[k]][0]
                synset_sublist.append(str(synset))
        context_dict[item] = context_sublist
        synset_dict[item] = synset_sublist
    return context_dict, synset_dict


def find_definition(lexical_item):
    synset_list = wn.synsets(lexical_item)
    for synset in synset_list:
        print(synset, synset.definition())


def create_seed_set(train_context_dict, train_dict, test_context_dict, test_dict):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_context_dict).toarray()
    # y_train = np.array(train_dict).astype('int')
    y_train = train_dict

    X_test = vectorizer.transform(test_context_dict).toarray()
    # y_test = np.array(test_dict).astype('int')
    y_test = test_dict
    return  X_train, y_train, X_test, y_test

def svm_classifier(X_train, y_train, X_test, y_test):
    clf = svm.SVC(kernel='linear')
    print("y_train", y_train)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    return accuracy_score(y_test, y_predict)

def random_forest_classifier(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    return accuracy_score(y_test, y_predict)


def boostrap_method(lexical_item, dev_instances, dev_key, test_instances, test_key):
    # Step 1: Find common words in dev and test set
    dev_lemma_list = []
    test_lemma_list = []
    for (dev, test) in zip(dev_instances.items(), test_instances.items()):
        dev_lemma_list.append(dev[1].lemma)
        test_lemma_list.append(test[1].lemma)
    dev_count = collections.Counter(dev_lemma_list)
    test_count = collections.Counter(test_lemma_list)

    test_mutual_count = {k: v for (k, v) in test_count.items() if k in dev_count}
    dev_mutual_count = {k: dev_count[k] for k in test_mutual_count}

    print(dev_mutual_count)
    print(test_mutual_count)

    # Step 2: Find the context of the chosen words and label the seed set
    dev_context_dict, dev_synset_dict = find_context(lexical_item, dev_instances, dev_key)
    print(dev_context_dict)
    print(dev_synset_dict)

    # Step 3: Give the correct labels for the words in the test set -> Will use it in the classifiers
    test_context_dict, test_synset_dict = find_context(lexical_item, test_instances, test_key)

    # Step 4: SVM Classifier -> Predict the sense

        # deal, lot, time : senseval3
        # year, time, group: senseval2

    for item in lexical_item:
        X_train, y_train, X_test, y_test = create_seed_set(dev_context_dict[item], dev_synset_dict[item],
                                                           test_context_dict[item], test_synset_dict[item])
        accuracy = svm_classifier(X_train, y_train, X_test, y_test)
        print("The accuracy of word \"{}\" is {}".format(item, accuracy))


def get_accuracy(predicted_key, expected_key):

    expected_key = {k: [wn.lemma_from_key(key).synset() for key in v][0] for k, v in expected_key.items()}
    num_key = len(expected_key)
    count = 0
    for predict, expect in zip(predicted_key.items(), expected_key.items()):
        if predict[1] == expect[1]:
            count+= 1
    accuracy = count/num_key

    return accuracy

if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = loader.load_instances(data_f)
    dev_key, test_key = loader.load_key(key_f)


    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}


    # print(dev_instances['d001.s016.t011'].context, dev_instances['d001.s016.t011'].id,
    #       dev_instances['d001.s016.t011'].lemma, dev_instances['d001.s016.t011'].index)

    dev_instances = data_preprocess(dev_instances)
    test_instances = data_preprocess(test_instances)

    # print(dev_instances['d001.s016.t011'].context, dev_instances['d001.s016.t011'].id,
    #       dev_instances['d001.s016.t011'].lemma, dev_instances['d001.s016.t011'].index)
    #
    # # Baseline
    # dev_baseline_predict = baseline(dev_instances)
    # test_baseline_predict = baseline(test_instances)
    # dev_baseline_accuracy = get_accuracy(dev_baseline_predict, dev_key)
    # test_baseline_accuracy = get_accuracy(test_baseline_predict, test_key)
    # print("Baseline dev accuracy is {}, and test accuracy is {}".format(dev_baseline_accuracy, test_baseline_accuracy))
    #
    # # lesk
    # dev_lesk_predict = lesk_wsd(dev_instances)
    # test_lesk_predict = lesk_wsd(test_instances)
    # dev_lesk_accuracy = get_accuracy(dev_lesk_predict, dev_key)
    # test_lesk_accuracy = get_accuracy(test_lesk_predict, test_key)
    # print("Lesk dev accuracy is {}, and test accuracy is {}".format(dev_lesk_accuracy, test_lesk_accuracy))

    # TODO : Deal with words that have understore

    # Bootstrapping Method 1
    lexical_item = ['year']
    # 'game', 'deal', 'year', 'lot', 'time', 'group'
    # time, deal


    # Add lexical items in senseval 2
    # data_f = 'senseval2.data.xml'
    # key_f = 'senseval2.gold.key.txt'
    # sense2_dev_instances = loader.senseval_load_instances(data_f)
    # sense2_dev_key = loader.senseval_load_key(key_f)
    #
    # sense2_dev_instances = {k: v for (k, v) in sense2_dev_instances.items() if k in sense2_dev_key}
    #
    # sense2_dev_instances = data_preprocess(sense2_dev_instances)
    #
    # sense2_dev_context_dict, sense2_dev_synset_dict = find_context(lexical_item, sense2_dev_instances, sense2_dev_key)
    #
    # print(sense2_dev_context_dict)
    # print(sense2_dev_synset_dict)
    #
    # # Add lexical items in senseval 3
    # data_f = 'senseval3.data.xml'
    # key_f = 'senseval3.gold.key.txt'
    # sense3_dev_instances = loader.senseval_load_instances(data_f)
    # sense3_dev_key = loader.senseval_load_key(key_f)
    #
    # sense3_dev_instances = {k: v for (k, v) in sense3_dev_instances.items() if k in sense3_dev_key}
    #
    # sense3_dev_instances = data_preprocess(sense3_dev_instances)
    #
    # sense3_dev_context_dict, sense3_dev_synset_dict = find_context(lexical_item, sense3_dev_instances, sense3_dev_key)
    #
    # print(sense3_dev_context_dict)
    # print(sense3_dev_synset_dict)


    boostrap_method(lexical_item, dev_instances, dev_key, test_instances, test_key)
    # print("Context for Each Lexical Term:")
    # for (k, v) in test_context_dict.items():
    #     print("-" * 40)
    #     print(k, ":")
    #     for context in v:
    #         print(context)
    #     print(k, " DEFINITIONS: ")
    #     find_definition(k)




