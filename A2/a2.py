import loader
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from collections import defaultdict
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from string import punctuation
from nltk.stem import WordNetLemmatizer


import xml.etree.cElementTree as ET
import codecs

def data_preprocess(instance):
    for (k, v) in instance.items():
        stop_words = set(stopwords.words("english"))
        stop_words.add('--')
        stop_words.add('``')
        stop_words.add('@card@')

        # remove stop word & punctuation & lower case
        v.context = [i.decode("utf-8") for i in v.context]
        v.context = [i for i in v.context if i not in punctuation]
        v.context = [i.lower() for i in v.context]
        v.context = [i for i in v.context if i not in stop_words]

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

    dev_instances = data_preprocess(dev_instances)
    test_instances = data_preprocess(test_instances)

    print(test_instances['d002.s001.t001'].context, test_instances['d002.s001.t001'].id,
          test_instances['d002.s001.t001'].lemma, test_instances['d002.s001.t001'].index)

    # Baseline
    dev_baseline_predict = baseline(dev_instances)
    test_baseline_predict = baseline(test_instances)
    dev_baseline_accuracy = get_accuracy(dev_baseline_predict, dev_key)
    test_baseline_accuracy = get_accuracy(test_baseline_predict, test_key)
    print("Baseline dev accuracy is {}, and test accuracy is {}".format(dev_baseline_accuracy, test_baseline_accuracy))

    # lesk
    dev_lesk_predict = lesk_wsd(dev_instances)
    test_lesk_predict = lesk_wsd(test_instances)
    dev_lesk_accuracy = get_accuracy(dev_lesk_predict, dev_key)
    test_lesk_accuracy = get_accuracy(test_lesk_predict, test_key)
    print("Lesk dev accuracy is {}, and test accuracy is {}".format(dev_lesk_accuracy, test_lesk_accuracy))



    # TODO : Deal with words that have understore
