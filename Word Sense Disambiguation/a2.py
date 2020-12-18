"""
COMP 550 Natural Language Processing
Assignment 2

Fulin Huang
260740689

"""

import loader
import nltk
import warnings
warnings.filterwarnings("ignore")
nltk.download('wordnet')
nltk.download('stopwords')
from collections import defaultdict
import collections
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def data_preprocess(instance):
    for (k, v) in instance.items():
        stop_words = set(stopwords.words("english"))
        # Remove strange symbols
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

# Lesk's Method
def lesk_wsd(instance):
    dict = {k: nltk.wsd.lesk(v.context, v.lemma.decode("utf-8")) for k, v in instance.items()}
    return dict

# Helper method for supervised model
def create_seed_set(type, train_context_dict, train_dict, test_context_dict, test_dict):
    if (type == 'tf'):
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    else :
        vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(train_context_dict).toarray()
    y_train = train_dict
    X_test = vectorizer.transform(test_context_dict).toarray()
    y_test = test_dict

    return  X_train, y_train, X_test, y_test

# Supervised model 1
def svm_classifier(X_train, y_train, X_test, y_test):
    print("-"*40)
    print("SVM Classifier:")
    param_grid = {'C':[0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1], 'kernel':['linear', 'rbf', 'sigmoid']}
    clf = GridSearchCV(svm.SVC(), param_grid)
    best_clf = clf.fit(X_train, y_train)
    y_predict = best_clf.predict(X_test)
    print("Best combination of hyperparameters for the SVM Classifier: ", clf.best_params_)

    return accuracy_score(y_test, y_predict)


# Supervised model 2
def logistic_regression_classifier(X_train, y_train, X_test, y_test):
    print("-"*40)
    print("LR Classifier:")
    param_grid = {'C':[0.01, 0.1, 1, 10, 100], 'penalty' :['none','l1', 'l2'], 'solver':['lbfgs', 'liblinear']}

    clf = GridSearchCV(LogisticRegression(max_iter=5000), param_grid)
    best_clf = clf.fit(X_train, y_train)
    y_predict = best_clf.predict(X_test)
    print("Best combination of hyperparameters for the LR Classifier", clf.best_params_)

    return accuracy_score(y_test, y_predict)

# Main function of supervised learning 1
def supervised_learning_1(type, dev_context, dev_synset, test_context, test_synset):

    X_train, y_train, X_test, y_test = create_seed_set(type, dev_context, dev_synset, test_context, test_synset)
    accuracy = svm_classifier(X_train, y_train, X_test, y_test)

    return accuracy

# Main function of supervised learning 2
def supervised_learning_2(type, dev_context, dev_synset, test_context, test_synset):

    X_train, y_train, X_test, y_test = create_seed_set(type, dev_context, dev_synset, test_context, test_synset)
    accuracy = logistic_regression_classifier(X_train, y_train, X_test, y_test)

    return accuracy

# Calculate accuracy (for baseline and lesk's method)
def get_accuracy(predicted_key, expected_key):

    expected_key = {k: [wn.lemma_from_key(key).synset() for key in v][0] for k, v in expected_key.items()}
    num_key = len(expected_key)
    count = 0
    for predict, expect in zip(predicted_key.items(), expected_key.items()):
        if predict[1] == expect[1]:
            count+= 1
    accuracy = count/num_key

    return accuracy

# Add all context/key to dictionary
def find_context_supervised(instance, key):
    context_list = []
    synset_list = []
    for (k, v) in instance.items():
        context = [i.decode("utf-8") for i in v.context]
        context = ' '.join(context)
        context_list.append(context)
        synset = [wn.lemma_from_key(i).synset() for i in key[k]][0]
        synset_list.append(str(synset))

    return context_list, synset_list

# Find context for a list of lexical item
# Return dictionary i.e. context = {'time': ['context1', 'context2'], 'case': ['context1', 'context2']}
def find_context(lexical_item, instance, key):
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
                synset_sublist.append(key[k][0])
        context_dict[item] = context_sublist
        synset_dict[item] = synset_sublist

    return context_dict, synset_dict

# Helper method: Check definition
def find_definition(lexical_item):
    synset_list = wn.synsets(lexical_item)
    for synset in synset_list:
        print(synset, synset.definition())

# Helper method: Find mutual words and count the number of occurrances in dev and test set
def find_mutual_words(dev_instances, test_instances):
    dev_lemma_list = []
    test_lemma_list = []

    for (k, v) in dev_instances.items():
        dev_lemma_list.append(v.lemma.decode("utf-8"))
    dev_count = collections.Counter(dev_lemma_list)

    for (k, v) in test_instances.items():
        test_lemma_list.append(v.lemma.decode("utf-8"))

    test_count = collections.Counter(test_lemma_list)

    test_mutual_count = {k: v for (k, v) in test_count.items() if k in dev_count}
    dev_mutual_count = {k: dev_count[k] for k in test_mutual_count}

    print("-"*40)
    print("Dev set has {} unique lemmas".format(len(dev_count)))
    print("Test set has {} unique lemmas".format(len(test_count)))
    print("There are {} mutual words in dev and test set".format(len(dev_mutual_count)))
    print("Mutual word count in the dev set: ", dev_mutual_count)
    print("Mutual word count in the test set: ", test_mutual_count)


# Part of the yarowsky's algorithm: final model for prediction
def final_predict(train_model, context, target, label, key):
    vectorizer = CountVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(target).toarray()
    model = train_model.fit(X_train, label)
    pred_list = []

    for sentence in context:

        X_test = vectorizer.transform(sentence.split()).toarray()
        test_predict = model.predict_proba(X_test)
        target_classA = test_predict[:, 0]
        class_idx = list(np.where(target_classA != 0.5)[0])
        if len(class_idx) != 0:
            avg = sum([target_classA[i] for i in class_idx]) / len(class_idx)
        else:
            avg = sum(target_classA)/len(target_classA)
        if avg >= 0.5:
            pred_list.append(key[0])
        else:
            pred_list.append(key[1])

    return pred_list

# Yarowsky's algorithm: continuously add confident confident outputs to the seed set
def yarowsky(train_model, context, seed_set, key, test_context):
    # initial seed set and label -> size increases over time
    target = seed_set
    label = ['A', 'B']

    # thresholds and # iterations
    thres_classA = 0.55
    thres_classB = 0.45
    iter = 12
    i = 0

    while i < iter and len(context) > 0:
        # Train a supervised learning algorithm from the seed set
        vectorizer = CountVectorizer(max_features=1000)
        X_train = vectorizer.fit_transform(target).toarray()
        model = train_model.fit(X_train, label)

        count = 0
        for sentence in context:
            token_list = sentence.split()
            count += len(token_list)

            # Apply the supervised model to the entire data set
            X_test = vectorizer.transform(sentence.split()).toarray()
            test_predict = model.predict_proba(X_test) # Get class probability
            target_class = test_predict[:,0]
            class_idx = list(np.where(target_class != 0.5)[0]) # words that are likely to be one of the class
            if (len(class_idx) != 0):
                avg = sum([target_class[i] for i in class_idx])/len(class_idx)
                # Keep the highly confident classification outputs to be the new seed set
                # else continue
                if (avg > thres_classA):
                    for token in token_list:
                        if(token not in target):
                            target.append(token)
                            label.append('A')
                    context.remove(sentence)

                elif (avg < thres_classB):
                    for token in token_list:
                        if(token not in target):
                            target.append(token)
                            label.append('B')
                    context.remove(sentence)
            # thres_classA += 0.005 # increase threshold gradually
            # thres_classB -= 0.005

        i+=1

    # Use the last model as the final model
    return final_predict(train_model, test_context, target, label, key)

def bootstrap_method(train_model, lexical_item, dev_instances, dev_key, test_instances, test_key, seed_dict, lexical_key):
    # Save dev context and key for the lexical items
    dev_context_dict, dev_synset_dict = find_context(lexical_item, dev_instances, dev_key)

    # Save test context and key for the lexical items
    test_context_dict, test_synset_dict = find_context(lexical_item, test_instances, test_key)

    # Concat context for doing yarowsky's algorithm
    concat_context_dict = dict.fromkeys(lexical_item)
    for key, value in dev_context_dict.items():
        concat_context_dict[key] = value+test_context_dict[key]

    total_correct_count = 0
    print("-----Bootstrap Method Accuracy-----")
    # evaluate each lexical item individually
    for item in lexical_item:
        output = yarowsky(train_model, concat_context_dict[item], seed_dict[item], lexical_key[item], test_context_dict[item])

        count = 0
        for idx, i in enumerate(output):
            if i == test_synset_dict[item][idx]:
                count += 1
                total_correct_count += 1
        print("Accuracy for lexical term \'{}\' is {}".format(item, count/len(test_synset_dict[item])))

    return total_correct_count

# Combine Lesk's algorithm and bootstrapping method (yarowsky) to evaluate the overall performance
def hybrid_system(train_model, test_instances, test_key, lexical_item, dev_instances, dev_key, seed_dict, lexical_key):
    count = 0
    for (instance, key) in zip(test_instances.items(), test_key.items()):
        v = instance[1]
        k = key[1]

        # Not in bootstrapping lexical items -> use lesk
        if(v.lemma.decode("utf-8")not in lexical_item):
            lesk_pred = nltk.wsd.lesk(v.context, v.lemma.decode("utf-8"))
            expected = [wn.lemma_from_key(key).synset() for key in k][0]
            if (lesk_pred == expected):
                count += 1

    # else: use bootstrapping method
    count += bootstrap_method(train_model, lexical_item, dev_instances, dev_key, test_instances, test_key, seed_dict, lexical_key)
    model_acc = count / len(test_instances)
    print("-----Hybrid System Accuracy (Lesk + Bootstrap)-----")
    print("Overall model accuracy is ", model_acc)


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = loader.load_instances(data_f)
    dev_key, test_key = loader.load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}
    print(len(dev_instances), len(test_instances))

    # data Preprocess
    dev_instances = data_preprocess(dev_instances)
    test_instances = data_preprocess(test_instances)

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

    dev_context, dev_synset = find_context_supervised(dev_instances, dev_key)
    test_context, test_synset = find_context_supervised(test_instances, test_key)

    # # Supervised Model 1
    # svm_accuracy = supervised_learning_1('count', dev_context, dev_synset, test_context, test_synset)
    # print("Accuracy using SVM Supervised Learning Model is ", svm_accuracy)

    # Supervised Model 2
    lr_accuracy = supervised_learning_2('count', dev_context, dev_synset, test_context, test_synset)
    print("Accuracy using Logic Regression Supervised Learning Model is ", lr_accuracy)

    # Bootstrapping Method
    # 1) Find mutual words in dev and test set
    find_mutual_words(dev_instances, test_instances)

    # 2) Choose 5 lexical items
    lexical_item = ['time', 'issue', 'action', 'united_states', 'case']

    # 3) Create initial seed set & keys
    seed_dict = {'time':['first', 'hour'], 'issue':['resolve', 'case'], 'action':['united_states','player'], 'united_states':['economy', 'country'], 'case':['clear', 'court']}
    lexical_key = {'time':['time%1:11:00::', 'time%1:28:05::'], 'issue':['issue%1:09:01::','issue%1:09:00::'], 'action':['action%1:04:04::','action%1:04:02::'], 'united_states':['united_states%1:14:00::','united_states%1:15:00::'], 'case':['case%1:11:00::', 'case%1:04:00::']}

    # 4) Build a hybrid system to evalute overall model performance (lesk + bootstrapping method)
        # ** Each lexical item is being evluated as well
    # train_model = svm.SVC(kernel='rbf',probability=True)
    train_model = LogisticRegression()
    hybrid_system(train_model, test_instances, test_key, lexical_item, dev_instances, dev_key, seed_dict, lexical_key)



