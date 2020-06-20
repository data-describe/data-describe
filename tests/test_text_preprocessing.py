import pytest
import pandas as pd
import nltk

from data_describe.text.text_preprocessing import (
    tokenize,
    to_lower,
    remove_digits,
    remove_punct,
    remove_single_char_and_spaces,
    remove_stopwords,
    preprocess_texts,
    bag_of_words_to_docs,
    create_doc_term_matrix,
    create_tfidf_matrix,
    ngram_freq,
)
from ._test_data import TEXT_DATA


@pytest.fixture
def data():
    return TEXT_DATA


@pytest.fixture
def tokenized_test_list_main():
    return tokenize(TEXT_DATA["test_list_main"])


def test_tokenizer(data):
    assert tokenize(data["test_list_main"]) == data["answer_key_tokenized"]


def test_to_lower(data, tokenized_test_list_main):
    assert to_lower(tokenized_test_list_main) == data["answer_key_lower"]


def test_remove_punct(data, tokenized_test_list_main):
    assert remove_punct(tokenized_test_list_main) == data["answer_key_remove_punct"]
    assert (
        remove_punct(tokenized_test_list_main, remove_all=True, replace_char="")
        == data["answer_key_remove_all_punct_no_space"]
    )
    assert (
        remove_punct(tokenized_test_list_main, remove_all=True)
        == data["answer_key_remove_all_punct_with_space"]
    )


def test_remove_digits(data):
    assert remove_digits(data["test_list_digits"]) == data["answer_key_remove_digits"]


def test_remove_single_char_and_spaces(data):
    assert (
        remove_single_char_and_spaces(data["test_list_single_char_and_spaces"])
        == data["answer_key_single_char_and_spaces"]
    )


def test_remove_stopwords(data, tokenized_test_list_main):
    assert (
        remove_stopwords(to_lower(tokenized_test_list_main))
        == data["answer_key_remove_stop_words"]
    )
    assert (
        remove_stopwords(
            to_lower(tokenized_test_list_main), more_words=data["more_words"]
        )
        == data["answer_key_remove_stop_words_more"]
    )


def test_lem_and_stem(data):
    assert (
        preprocess_texts(data["test_list_lem_and_stem"], lem=True)
        == data["answer_key_lem"]
    )
    assert (
        preprocess_texts(data["test_list_lem_and_stem"], stem=True)
        == data["answer_key_stem"]
    )


def test_bag_of_words(data):
    bag = bag_of_words_to_docs(preprocess_texts(data["test_list_main"]))

    assert len(data["test_list_main"]) == len(bag)
    assert isinstance(bag, list)
    assert isinstance(bag[0], str)


def test_matrices_length(data):
    tfidf_matrix = create_tfidf_matrix(data["test_list_main"])
    doc_term_matrix = create_doc_term_matrix(data["test_list_main"])

    assert isinstance(tfidf_matrix, pd.DataFrame)
    assert len(tfidf_matrix) == len(data["test_list_main"])
    assert isinstance(doc_term_matrix, pd.DataFrame)
    assert len(doc_term_matrix) == len(data["test_list_main"])


def test_length_and_type(data):
    run_preprocessing = preprocess_texts(data["test_list_main"])

    assert len(data["test_list_main"]) == len(run_preprocessing)
    assert isinstance(run_preprocessing, list)
    assert isinstance(run_preprocessing[0], list)
    assert isinstance(run_preprocessing[0][0], str)


def test_custom_pipeline(data):
    def shout(text_docs_bow):
        return [[word.upper() for word in doc] for doc in text_docs_bow]

    assert (
        preprocess_texts(data["test_list_custom"], custom_pipeline=["tokenize", shout])
        == data["answer_key_custom"]
    )


def test_ngrams(data):
    n = 4
    n_grams = ngram_freq(data["test_list_main"], n)
    assert isinstance(n_grams, nltk.FreqDist)
