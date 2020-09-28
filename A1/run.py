import argparse

from preprocessing import preprocessing
# from classification import classification


# def classification():


def main(args):

    if args.feature_extraction:
        preprocess_class = preprocessing()
        corpus, label = preprocess_class.data_preprocess()
        # preprocess_class.remove_infrequent_word(corpus)
        new_corpus = [" ".join(preprocess_class.get_features(sentence, "stopword")) for sentence in corpus] # if include stopword
        new_corpus = [" ".join(preprocess_class.get_features(sentence, "stem")) for sentence in new_corpus]
        new_corpus = [" ".join(preprocess_class.get_features(sentence, "lemmatize")) for sentence in new_corpus]



    # if args.classification:
        # classify_class = classification()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run data preprocessing and perform classification')

    parser.add_argument('--data', dest='feature_extraction', action='store_true', default=False, help='Start data preprocess')
    parser.add_argument('--classify', dest='classification', action='store_true', default=False, help='Perform classification')


    args = parser.parse_args()
    main(args)