import argparse

from preprocessing import preprocessing
# from classification import classification


# def classification():


def main(args):

    if args.feature_extraction:
        preprocess_class = preprocessing()
        preprocess_class.data_preprocess()

    # if args.classification:
        # classify_class = classification()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run data preprocessing and perform classification')

    parser.add_argument('--data', dest='feature_extraction', action='store_true', default=False, help='Start data preprocess')
    parser.add_argument('--classify', dest='classification', action='store_true', default=False, help='Perform classification')


    args = parser.parse_args()
    main(args)