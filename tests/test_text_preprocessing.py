from sklearn.datasets import fetch_20newsgroups
from mwdata.text.text_preprocessing import *
import random
import pytest


@pytest.fixture
def get_data():
    docs = ["This is an article talking about Spider-man. Because Spider-man is cool.",
            "Spiders are one of the weirdest things on earth. Man, I'd hate to live with spiders.",
            "James Bond is the best spy that ever lived!",
            "Sometimes people just run out of things to say. It's life, we gotta move on."]
    return docs


def test_tokenizer():
    test_list = ['This is a list to test my tokenizer function.', 'Test this please']
    answer_key = [['This', 'is', 'a', 'list', 'to', 'test', 'my', 'tokenizer', 'function', '.'],
                  ['Test', 'this', 'please']]

    assert tokenize(test_list) == answer_key


def test_to_lower():
    test_list = ['THIS IS a TEst TO SEE if my LOWer FUNCTION works', 'still testing', 'TESTING THIS TOO']
    answer_key = [['this', 'is', 'a', 'test', 'to', 'see', 'if', 'my', 'lower', 'function', 'works'],
                  ['still', 'testing'], ['testing', 'this', 'too']]
    test_answer = to_lower(tokenize(test_list))

    assert test_answer == answer_key


def test_remove_punct():
    test_list = [['this.', 'is', 'a&', 'te?st', '.', 'thank-you', '!eat']]
    answer_key = [['this', 'is', 'a', 'te?st', 'thank-you', 'eat']]
    answer_key_remove_all_no_space = [['this', 'is', 'a', 'test', '', 'thankyou', 'eat']]
    answer_key_remove_all_space = [['this', 'is', 'a', 'te', 'st', 'thank', 'you', 'eat']]
    test_answer = remove_punct(test_list)
    test_answer_remove_all_no_space = remove_punct(test_list, remove_all=True, replace_char='')
    test_answer_remove_all_space = remove_punct(test_list, remove_all=True)

    assert answer_key == test_answer
    assert answer_key_remove_all_no_space == test_answer_remove_all_no_space
    assert answer_key_remove_all_space == test_answer_remove_all_space


def test_remove_digits():
    test_list = [['th3is', 'is', '8', 'a', 'd1s4ster', 'but', 'no', 'numb3rs']]
    answer_key = [['', 'is', '', 'a', '', 'but', 'no', '']]
    test_answer = remove_digits(test_list)

    assert test_answer == answer_key


def test_remove_single_char_and_spaces():
    test_list = [['a', ' ', 'nice', '    ', 'day', 'to', 'b', 'outside', 'thank you']]
    answer_key = [['nice', 'day', 'to', 'outside', 'thank', 'you']]
    test_answer = remove_single_char_and_spaces(test_list)

    assert answer_key == test_answer


def test_remove_stopwords():
    test_list = [['please', 'do', 'not', 'eat', 'the', 'rest', 'of', 'my', 'pineapple', 'pizza']]
    answer_key = [['please', 'eat', 'rest', 'pineapple', 'pizza']]
    answer_key_more = [['please', 'rest', 'pineapple']]
    test_answer = remove_stopwords(test_list)
    test_answer_more = remove_stopwords(test_list, more_words=['eat', 'pizza'])

    assert answer_key == test_answer
    assert answer_key_more == test_answer_more


def test_lem_and_stem(get_data):
    test_docs = [
        'Mars is the greatest planet to start terraforming; it would be amazing to see geese flying on the surface!']
    docs_lem = preprocess_texts(test_docs, lem=True)
    docs_stem = preprocess_texts(test_docs, stem=True)

    assert docs_lem == [
        ['mar', 'greatest', 'planet', 'start', 'terraforming', 'would', 'amazing', 'see', 'goose', 'flying', 'surface']]
    assert docs_stem == [
        ['mar', 'greatest', 'planet', 'start', 'terraform', 'would', 'amaz', 'see', 'gees', 'fly', 'surfac']]


def test_bag_of_words(get_data):
    bag = bag_of_words_to_docs(preprocess_texts(get_data))

    assert len(get_data) == len(bag)
    assert isinstance(bag, list)
    assert isinstance(bag[0], str)


def test_matrices_length(get_data):
    tfidf_matrix = create_tfidf_matrix(get_data)
    doc_term_matrix = create_doc_term_matrix(get_data)

    assert isinstance(tfidf_matrix, pd.DataFrame)
    assert len(tfidf_matrix) == len(get_data)
    assert isinstance(doc_term_matrix, pd.DataFrame)
    assert len(doc_term_matrix) == len(get_data)


def test_length_and_type(get_data):
    run_preprocessing = preprocess_texts(get_data)

    assert len(get_data) == len(run_preprocessing)
    assert isinstance(run_preprocessing, list)
    assert isinstance(run_preprocessing[0], list)
    assert isinstance(run_preprocessing[0][0], str)


def test_custom_pipeline():
    def shout(text_docs_bow):
        return [[word.upper() for word in doc] for doc in text_docs_bow]

    test_list = ['this is an absolutely phenomenal day.', 'I love CHICKEN']
    answer_key = [['THIS', 'IS', 'AN', 'ABSOLUTELY', 'PHENOMENAL', 'DAY', '.'], ['I', 'LOVE', 'CHICKEN']]
    test_answer = preprocess_texts(test_list, custom_pipeline=['tokenize', shout])

    assert test_answer == answer_key


def test_ngrams(get_data):
    n = 4
    n_grams = ngram_freq(get_data, n)
    assert isinstance(n_grams, nltk.FreqDist)
