from nltk import tokenize
import string


def load(file_path='../data/politics.txt'):
    with open(file_path) as f:
        return f.read()


# region Preprocessing
def load_sentences(file_path):
    with open(file_path) as f:
        text = f.read()
    sentences = parse_sentences(text)
    return sentences


def load_parsed_sentences(file_path):
    return [parse_words(sentence) for sentence in load_sentences(file_path)]


def simplify(s: str):
    s = s.replace('-', ' ')
    s = s.replace('\n', ' ')
    return s.lower().translate(str.maketrans('', '', string.punctuation))


def parse_sentences(text):
    return [simplify(sentence) for sentence in
            tokenize.sent_tokenize(text)]


def parse_words(sentence: str):
    return tokenize.word_tokenize(sentence)


# endregion


if __name__ == '__main__':
    corpus = load_parsed_sentences('../data/politics.txt')
    generated_sentences = load_parsed_sentences('../data/text.txt')
    print(generated_sentences)
