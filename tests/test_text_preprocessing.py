import itertools

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


@pytest.fixture
def tokenized_test_list_main(text_data):
    return list(itertools.chain.from_iterable(tokenize(text_data["test_list_main"])))


def test_tokenizer(text_data):
    assert (
        list(itertools.chain.from_iterable(tokenize(text_data["test_list_main"]))) == text_data["answer_key_tokenized"]
    )


def test_to_lower(text_data, tokenized_test_list_main):
    assert [list(generator) for generator in list(itertools.chain.from_iterable(to_lower(tokenized_test_list_main)))] == text_data["answer_key_lower"]


def test_remove_punct(text_data, tokenized_test_list_main):
    assert (
        [list(generator) for generator in list(itertools.chain.from_iterable(remove_punct(tokenized_test_list_main)))]
        == text_data["answer_key_remove_punct"]
    )
    assert (
        [list(generator) for generator in list(itertools.chain.from_iterable(remove_punct(tokenized_test_list_main, remove_all=True, replace_char="")))]
        == text_data["answer_key_remove_all_punct_no_space"]
    )
    assert (
        [list(generator) for generator in list(itertools.chain.from_iterable(remove_punct(tokenized_test_list_main, remove_all=True)))]
        == text_data["answer_key_remove_all_punct_with_space"]
    )


def test_remove_digits(text_data):
    assert (
        [list(generator) for generator in list(itertools.chain.from_iterable(remove_digits(text_data["test_list_digits"])))]
        == text_data["answer_key_remove_digits"]
    )


def test_remove_single_char_and_spaces(text_data):
    assert (
        [list(generator) for generator in list(itertools.chain.from_iterable(remove_single_char_and_spaces(text_data["test_list_single_char_and_spaces"])))]
        == text_data["answer_key_single_char_and_spaces"]
    )


def test_remove_stopwords(text_data, tokenized_test_list_main):
    assert (
        [list(generator) for generator in list(itertools.chain.from_iterable(remove_stopwords(
            [list(generator) for generator in list(itertools.chain.from_iterable(to_lower(tokenized_test_list_main)))],
        )))]
        == text_data["answer_key_remove_stop_words"]
    )
    assert (
        [list(generator) for generator in list(itertools.chain.from_iterable(remove_stopwords(
            [list(generator) for generator in list(itertools.chain.from_iterable(to_lower(tokenized_test_list_main)))],
            more_words=text_data["more_words"]
        )))]
        == text_data["answer_key_remove_stop_words_more"]
    )


def test_lem_and_stem(text_data):
    assert (
        preprocess_texts(text_data["test_list_lem_and_stem"], lem=True)
        == text_data["answer_key_lem"]
    )
    assert (
        preprocess_texts(text_data["test_list_lem_and_stem"], stem=True)
        == text_data["answer_key_stem"]
    )


def test_bag_of_words(text_data):
    bag = list(itertools.chain.from_iterable(bag_of_words_to_docs(preprocess_texts(text_data["test_list_main"]))))

    assert len(text_data["test_list_main"]) == len(bag)
    assert isinstance(bag, list)
    assert isinstance(bag[0], str)


def test_matrices_length(text_data):
    tfidf_matrix = create_tfidf_matrix(text_data["test_list_main"])
    doc_term_matrix = create_doc_term_matrix(text_data["test_list_main"])

    assert isinstance(tfidf_matrix, pd.DataFrame)
    assert len(tfidf_matrix) == len(text_data["test_list_main"])
    assert isinstance(doc_term_matrix, pd.DataFrame)
    assert len(doc_term_matrix) == len(text_data["test_list_main"])


def test_length_and_type(text_data):
    run_preprocessing = preprocess_texts(text_data["test_list_main"])

    assert len(text_data["test_list_main"]) == len(run_preprocessing)
    assert isinstance(run_preprocessing, list)
    assert isinstance(run_preprocessing[0], list)
    assert isinstance(run_preprocessing[0][0], str)


def test_custom_pipeline(text_data):
    def shout(text_docs_bow):
        return [[word.upper() for word in doc] for doc in text_docs_bow]

    assert (
        preprocess_texts(
            text_data["test_list_custom"], custom_pipeline=["tokenize", shout]
        )
        == text_data["answer_key_custom"]
    )


def test_ngrams(text_data):
    n = 4
    n_grams = ngram_freq(text_data["test_list_main"], n)
    assert isinstance(n_grams, nltk.FreqDist)
