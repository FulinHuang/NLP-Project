from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from utils import load_parsed_sentences, load_sentences, load, simplify
from rouge import Rouge, FilesRouge


def bleu_score(corpus_path, generated_sentences_path, n=4):
    corpus = load_parsed_sentences(corpus_path)
    generated_sentences = load_parsed_sentences(generated_sentences_path)
    print(len(corpus), corpus[:3])
    print(len(generated_sentences), generated_sentences[:3])
    score = corpus_bleu([corpus for _ in generated_sentences], generated_sentences, weights=[1 for _ in range(n)])
    print(score)


def rouge_score(corpus_path, generated_sentences_path):
    corpus = simplify(load(corpus_path))
    generated_sentences = simplify(load(generated_sentences_path))
    print(len(corpus), len(corpus.split()), repr(corpus[:300]))
    print(len(generated_sentences), len(generated_sentences.split()), generated_sentences[:300])
    rouge = Rouge()
    scores = rouge.get_scores(generated_sentences, corpus)
    print(scores)


def rouge_score_s(corpus_path, generated_sentences_path):
    corpus = load_sentences(corpus_path)
    generated_sentences = load_sentences(generated_sentences_path)
    print(len(corpus), corpus[:3])
    print(len(generated_sentences), generated_sentences[:3])
    rouge = Rouge()
    for sentence in generated_sentences:
        for ref in corpus:
            print(sentence, ref)
            scores = rouge.get_scores(sentence, ref)
            rouge.get_scores(sentence, ref)
            print(scores)


def main_bleu():
    corpus_path = '../data/politics.txt'
    generated_names = ['word_lstm_output_generatedText.txt']
    ns = [1, 2, 3, 4]
    for name in generated_names:
        for n in ns:
            print(name, n)
            bleu_score(corpus_path, '../data/' + name, n)


def main_rouge():
    corpus_path = '../data/politics_small.txt'
    generated_path = '../data/word_lstm_output_generatedText.txt'
    rouge_score(corpus_path, generated_path)


if __name__ == '__main__':
    main_bleu()
