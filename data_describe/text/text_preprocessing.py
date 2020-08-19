"""Text preprocessing module.

This module contains a number of methods by which text documents can be preprocessed. The individual preprocessing functions can be classified as "Bag of Words Functions"
(to_lower, remove_punct, remove_digits, remove_single_char_and_spaces, remove_stopwords, lemmatize, stem) or "Document Functions" (tokenize, bag_of_words_to_docs).
Each of the functions in these groups return generator objects, and when using them on their own, the internal function to_list can be utilized as depicted below.

Example:
    Individual Document Functions should be processed as such::

        tokenized_docs = to_list(tokenize(original_docs), bow=False)

    Individual Bag of Words Functions should be processed as such::

        lower_case_docs_bow = to_list(to_lower(original_docs_bow))
"""

import re
import string
import warnings
from types import GeneratorType
from typing import List, Optional, Any, Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from data_describe.text import text_preprocessing
from data_describe.compat import requires, _compat

warnings.filterwarnings("ignore", category=UserWarning, module="gensim")


@requires("nltk")
def tokenize(text_docs: Iterable[str]) -> Iterable[Iterable[str]]:
    """Turns list of documents into "bag of words" format.

    Args:
        text_docs: A list of text documents in string format

    Returns:
        A generator expression for all of the processed documents
    """
    return (_compat.word_tokenize(doc) for doc in text_docs)


def to_lower(text_docs_bow: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
    """Converts all letters in documents ("bag of words" format) to lowercase.

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        A generator expression for all of the processed documents
    """
    return ((word.lower() for word in doc) for doc in text_docs_bow)


def remove_punct(
    text_docs_bow: Iterable[Iterable[str]],
    replace_char: str = "",
    remove_all: bool = False,
) -> Iterable[Iterable[str]]:
    """Removes all instances of punctuation from documents (e.g. periods, question marks, etc.).

    Args:
        text_docs_bow: A list of lists of words from a document
        replace_char: Character to replace punctuation instances with. Default is space
        remove_all: If True, removes all instances of punctuation from document. Default is False, which only removes
        leading and/or trailing instances

    Returns:
        A generator expression for all of the processed documents
    """
    if remove_all:
        new_docs = (
            (re.sub(r"[^\w\s]|_", replace_char, word) for word in doc)
            for doc in text_docs_bow
        )
        return (
            (
                word
                for word in doc
                if not (word in string.punctuation or word.strip() == "")
            )
            for doc in new_docs
        )
    else:
        new_docs = (
            (re.sub(r"^([^\w\s]|_)?(.+?)([^\w\s]|_)?$", r"\2", word) for word in doc)
            for doc in text_docs_bow
        )
        return (
            (
                word
                for word in doc
                if not (len(word) == 1 and word in string.punctuation)
            )
            for doc in new_docs
        )


def remove_digits(text_docs_bow: Iterable[Iterable[str]],) -> Iterable[Iterable[str]]:
    """Removes all numbers and words containing numerical digits from documents.

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        A generator expression for all of the processed documents
    """
    return ((re.sub(r"\w*\d\w*", "", word) for word in doc) for doc in text_docs_bow)


def remove_single_char_and_spaces(
    text_docs_bow: Iterable[Iterable[str]],
) -> Iterable[Iterable[str]]:
    """Removes all words that contain only one character and blank spaces from documents.

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        A generator expression for all of the processed documents
    """
    new_docs = []
    for doc in text_docs_bow:
        new_doc = []
        for word in doc:
            new_word = word.strip()
            if " " in new_word:
                new_words = new_word.split()
                for item in new_words:
                    new_doc.append(item)
            elif len(new_word) > 1:
                new_doc.append(new_word)
        new_docs.append(new_doc)
    return (doc for doc in new_docs)


@requires("nltk")
def remove_stopwords(
    text_docs_bow: Iterable[Iterable[str]], custom_stopwords: Optional[List[str]] = None
) -> Iterable[Iterable[str]]:
    """Removes all "stop words" from documents. "Stop words" can be defined as commonly used words which are typically useless for NLP.

    Args:
        text_docs_bow: A list of lists of words from a document
        custom_stopwords: An optional list of words to remove along with the stop words. Default is None

    Returns:
        A generator expression for all of the processed documents
    """
    stop_words_original = set(_compat.stopwords.words("english"))

    if custom_stopwords:
        stop_words = stop_words_original.union(custom_stopwords)
    else:
        stop_words = stop_words_original

    return ((word for word in doc if word not in stop_words) for doc in text_docs_bow)


@requires("nltk")
def lemmatize(text_docs_bow: Iterable[Iterable[str]],) -> Iterable[Iterable[str]]:
    """Lemmatizes all words in documents. Lemmatization is grouping words together by their reducing them to their inflected forms so they can be analyzed as a single item.

    Args:
        text_docs_bow: A lists of list of words from a document

    Returns:
        A generator expression for all of the processed documents
    """
    lemmatizer = compat.WordNetLemmatizer()
    return ((lemmatizer.lemmatize(word) for word in doc) for doc in text_docs_bow)


@requires("nltk")
def stem(text_docs_bow: Iterable[Iterable[str]],) -> Iterable[Iterable[str]]:
    """Stems all words in documents. Stemming is grouping words together by taking the stems of their inflected forms so they can be analyzed as a single item.

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        A generator expression for all of the processed documents
    """
    stemmer = _compat.LancasterStemmer()
    return ((stemmer.stem(word) for word in doc) for doc in text_docs_bow)


def bag_of_words_to_docs(text_docs_bow: Iterable[Iterable[str]]) -> Iterable[str]:
    """Converts list of documents in "bag of words" format back into form of document being stored in one string.

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        A generator expression for all of the processed documents
    """
    return (" ".join(doc) for doc in text_docs_bow)


def create_tfidf_matrix(text_docs: Iterable[str]) -> pd.DataFrame:
    """Creates a Term Frequency-Inverse Document Frequency matrix.

    Args:
        text_docs: A list of strings of text documents

    Returns:
        matrix_df: Pandas DataFrame of TF-IDF matrix with documents as rows and words as columns
    """
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(text_docs).toarray()
    matrix_df = pd.DataFrame(matrix, columns=tfidf.get_feature_names())
    return matrix_df


def create_doc_term_matrix(text_docs: Iterable[str]) -> pd.DataFrame:
    """Creates a document-term matrix which gives wordcount per document.

    Args:
        text_docs: A list of strings of text documents

    Returns:
        matrix_df: Pandas DataFrame of document-term matrix with documents as rows and words as columns
    """
    countvec = CountVectorizer()
    matrix = countvec.fit_transform(text_docs).toarray()
    matrix_df = pd.DataFrame(matrix, columns=countvec.get_feature_names())
    return matrix_df


def preprocess_texts(
    text_docs: Iterable[str],
    lem: bool = False,
    stem: bool = False,
    custom_pipeline: List = None,
) -> Iterable[Any]:
    """Cleans list of documents by running through a customizable text-preprocessing pipeline.

    Args:
        text_docs: A list of strings of text documents (also accepts arrays and Pandas series)
        lem: If True, lemmatization becomes part of the pre-processing. Recommended to set as False and run
        user-created lemmatization function if pipeline is customized. Default is False
        stem: If True, stemming becomes part of the pre-processing. Recommended to set as False and run
        user-created stemming function if pipeline is customized. Default is False
        custom_pipeline: A custom list of strings and/or function objects which are the function names that
        text_docs_bow will run through. Default is None, which uses the pipeline:
        ['tokenize', 'to_lower', 'remove_punct', 'remove_digits', 'remove_single_char_and_spaces', 'remove_stopwords']

    Returns:
        text_docs: List of lists of words for each document which have undergone a pre-processing pipeline
    """
    if not custom_pipeline:
        pipeline = [
            "tokenize",
            "to_lower",
            "remove_punct",
            "remove_digits",
            "remove_single_char_and_spaces",
            "remove_stopwords",
        ]
    else:
        pipeline = custom_pipeline

    if lem:
        pipeline.append("lemmatize")
    if stem:
        pipeline.append("stem")

    for function in pipeline:
        if isinstance(function, str):
            current_method = getattr(text_preprocessing, function)
            text_docs = current_method(text_docs)
        else:
            text_docs = function(text_docs)

    return to_list(text_docs)


def to_list(text_docs_gen) -> List[Any]:
    """Converts a generator expression from an individual preprocessing function into a list.

    Args:
        text_docs_gen: A generator expression for the processed text documents

    Returns:
        A list of processed text documents or a list of tokens (list of strings) for each document
    """
    if not isinstance(text_docs_gen, GeneratorType):
        return text_docs_gen
    else:
        return [to_list(i) for i in text_docs_gen]


@requires("nltk")
def ngram_freq(
    text_docs_bow: Iterable[Iterable[str]], n: int = 3, only_n: bool = False
) -> compat.FreqDist:
    """Generates frequency distribution of "n-grams" from all of the text documents.

    Args:
        text_docs_bow: A list of lists of words from a document
        n: Highest 'n' for n-gram sequence to include. Default is 3
        only_n: If True, will only include n-grams for specified value of 'n'. Default is False, which also includes
        n-grams for all numbers leading up to 'n'

    Returns:
        freq: Dictionary which contains all identified n-grams as keys and their respective counts as their values
    """
    if n < 2:
        raise ValueError("'n' must be a number 2 or greater")

    freq = _compat.FreqDist()
    for doc in text_docs_bow:
        if only_n:
            current_ngrams = compat.ngrams(doc, n)
            freq.update(current_ngrams)
        else:
            for num in range(2, n + 1):
                current_ngrams = compat.ngrams(doc, num)
                freq.update(current_ngrams)
    return freq


@requires("gensim")
def filter_dictionary(text_docs: List[str], no_below: int = 10, no_above: float = 0.2):
    """Filters words that appear less than a certain amount of times in the document and returns a Gensim.

    Args:
        text_docs: A list of list of words from a document, can include n-grams up to 3
        no_below: Keep tokens which are contained in at least no_below documents. Default is 10
        no_above: Keep tokens which are contained in no more than no_above portion of documents (fraction of total
        corpus size). Default is 0.2

    Returns:
        dictionary: Gensim Dictionary encapsulates the mapping between normalized words and their integer ids
        corpus: Bag of Words (BoW) representation of documents (token_id, token_count)
    """
    dictionary = _compat.Dictionary(text_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(doc) for doc in text_docs]
    return dictionary, corpus
