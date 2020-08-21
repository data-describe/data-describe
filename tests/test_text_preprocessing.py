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
    to_list,
    bag_of_words_to_docs,
    create_doc_term_matrix,
    create_tfidf_matrix,
    ngram_freq,
)


@pytest.fixture
def tokenized_test_list_main(text_data):
    return to_list(tokenize(text_data["test_list_main"]), bow=False)


def test_tokenizer(text_data):
    assert (
        to_list(tokenize(text_data["test_list_main"]), bow=False)
        == text_data["answer_key_tokenized"]
    ), "Output did not correctly tokenize documents"


def test_to_lower(text_data, tokenized_test_list_main):
    assert (
        to_list(to_lower(tokenized_test_list_main)) == text_data["answer_key_lower"]
    ), "Output did not correctly convert documents to lowercase"


def test_remove_punct(text_data, tokenized_test_list_main):
    assert (
        to_list(remove_punct(tokenized_test_list_main))
        == text_data["answer_key_remove_punct"]
    ), "Output did not correctly remove punctuation inside of words from documents and replace them with spaces"
    assert (
        to_list(
            remove_punct(tokenized_test_list_main, remove_all=True, replace_char="")
        )
        == text_data["answer_key_remove_all_punct_no_space"]
    ), "Output did not correctly remove all punctuation from documents and replace them with nothing"
    assert (
        to_list(remove_punct(tokenized_test_list_main, remove_all=True))
        == text_data["answer_key_remove_all_punct_with_space"]
    ), "Output did not correctly remove all punctuation from documents and replace them with spaces"


def test_remove_digits(text_data):
    assert (
        to_list(remove_digits(text_data["test_list_digits"]))
        == text_data["answer_key_remove_digits"]
    ), "Output did not correctly remove all digits from documents"


def test_remove_single_char_and_spaces(text_data):
    assert (
        to_list(
            remove_single_char_and_spaces(text_data["test_list_single_char_and_spaces"])
        )
        == text_data["answer_key_single_char_and_spaces"]
    ), "Output did not correctly remove all instance of single character words and spaces from documents"


def test_remove_stopwords(text_data, tokenized_test_list_main):
    assert (
        to_list(remove_stopwords(to_list(to_lower(tokenized_test_list_main))))
        == text_data["answer_key_remove_stop_words"]
    ), "Output did not correctly remove NLTK stopwords from documents"
    assert (
        to_list(
            remove_stopwords(
                to_list(to_lower(tokenized_test_list_main)),
                more_words=text_data["more_words"],
            )
        )
        == text_data["answer_key_remove_stop_words_more"]
    ), "Output did not correctly remove NLTK stopwords and custom added stopwords from documents"


def test_lem_and_stem(text_data):
    assert (
        preprocess_texts(text_data["test_list_lem_and_stem"], lem=True)
        == text_data["answer_key_lem"]
    ), "Output did not correctly lemmatize words in documents"
    assert (
        preprocess_texts(text_data["test_list_lem_and_stem"], stem=True)
        == text_data["answer_key_stem"]
    ), "Output did not correctly stem words in documents"


def test_bag_of_words(text_data):
    bag = to_list(
        bag_of_words_to_docs(preprocess_texts(text_data["test_list_main"])), bow=False
    )

    assert len(text_data["test_list_main"]) == len(
        bag
    ), "Number of documents in input does not match number of documents in output"
    assert isinstance(bag, list), "Output is not of the expected return type of list"
    assert isinstance(
        bag[0], str
    ), "Output does not contain expected type of string inside of return value"


def test_matrices_length(text_data):
    tfidf_matrix = create_tfidf_matrix(text_data["test_list_main"])
    doc_term_matrix = create_doc_term_matrix(text_data["test_list_main"])

    assert isinstance(
        tfidf_matrix, pd.DataFrame
    ), "Output is not of the expected return type of Pandas data frame"
    assert len(tfidf_matrix) == len(
        text_data["test_list_main"]
    ), "Number of documents in output does not match number of documents in input"
    assert isinstance(
        doc_term_matrix, pd.DataFrame
    ), "Output is not of the expected return type of Pandas data frame"
    assert len(doc_term_matrix) == len(
        text_data["test_list_main"]
    ), "Number of documents in output does not match number of documents in input"


def test_length_and_type(text_data):
    run_preprocessing = preprocess_texts(text_data["test_list_main"])

    assert len(text_data["test_list_main"]) == len(
        run_preprocessing
    ), "Number of documents in output does not match number of documents in input"
    assert isinstance(
        run_preprocessing, list
    ), "Output is not of the expected return type of list"
    assert isinstance(
        run_preprocessing[0], list
    ), "Output does not contain expected type of list inside of return value"
    assert isinstance(
        run_preprocessing[0][0], str
    ), "Output does not contain expected type of string inside of return value inside of the return value"


def test_custom_pipeline(text_data):
    def shout(text_docs_bow):
        return [[word.upper() for word in doc] for doc in text_docs_bow]

    assert (
        preprocess_texts(
            text_data["test_list_custom"], custom_pipeline=["tokenize", shout]
        )
        == text_data["answer_key_custom"]
    ), "Output did not correctly incorporate custom function into preprocessing pipeline"


def test_ngrams(text_data):
    n = 4
    n_grams = ngram_freq(text_data["test_list_main"], n)
    assert isinstance(
        n_grams, nltk.FreqDist
    ), "Output is not of the expected return type of NLTK Frequency Distribution object"
