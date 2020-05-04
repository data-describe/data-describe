import pandas as pd
import re
import string
import mwdata
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams

try:
    word_tokenize('try this')
except LookupError:
    nltk.download('punkt')

try:
    stop_words_original = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')


def tokenize(text_docs):
    """Turns list of documents into "bag of words" format

    Args:
        text_docs: A list of text documents in string format
    
    Returns:
        List of lists of words for each text document

    """
    new_text_docs_bow = [word_tokenize(doc) for doc in text_docs]
    return new_text_docs_bow


def to_lower(text_docs_bow):
    """Converts all letters in documents ("bag of words" format) to lowercase

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        List of lists of lowercase words for each text document

    """
    new_text_docs_bow = [[word.lower() for word in doc] for doc in text_docs_bow]

    return new_text_docs_bow


def remove_punct(text_docs_bow, replace_char=" ", remove_all=False):
    """Removes all punctuation from documents

    Args:
        text_docs_bow: A list of lists of words from a document
        replace_char: Character to replace puncutation instances with. Default is space.
        remove_all: If True, removes all instances of punctuation from document. Otherwise, just removes leading
                    and/or trailing instances

    Returns:
        List of lists of words for each text document without punctuation

    """
    if remove_all:
        new_text_docs_bow = [[re.sub('[^\w\s]|_', replace_char, word) for word in doc]
                             for doc in text_docs_bow]

        # Split up any tokens with spaces added now
        new_text_docs_bow = [[tokenize([word])[0] if " " in word else [word] for word in doc]
                             for doc in new_text_docs_bow]

        # Flatten nested lists
        new_text_docs_bow = [[item for sublist in doc for item in sublist]
                             for doc in new_text_docs_bow]
    else:
        new_text_docs_bow = [[re.sub('^([^\w\s]|_)?(.+?)([^\w\s]|_)?$', r'\2', word) for word in doc]
                             for doc in text_docs_bow]

    new_text_docs_bow_final = [[word for word in doc if not (len(word) == 1 and word in string.punctuation)]
                               for doc in new_text_docs_bow]

    return new_text_docs_bow_final


def remove_digits(text_docs_bow):
    """Removes all numbers and words containing digits from documents

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        List of lists of words for each text documents without numbers or words containing numbers

    """
    new_text_docs_bow = [[re.sub('\w*\d\w*', '', word) for word in doc] for doc in text_docs_bow]

    return new_text_docs_bow


def remove_single_char_and_spaces(text_docs_bow):
    """Removes all single character words and spaces from documents

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        List of lists of words for each text document without single character words

    """
    new_text_docs_bow = []
    for doc in text_docs_bow:
        new_word_list = []
        for word in doc:
            new_word = word.strip()
            if ' ' in new_word:
                new_words = new_word.split()
                for item in new_words:
                    new_word_list.append(item)
            elif len(new_word) > 1:
                new_word_list.append(new_word)
        new_text_docs_bow.append(new_word_list)
    return new_text_docs_bow


def remove_stopwords(text_docs_bow, more_words=None):
    """Removes all "stop words" from documents

    Args:
        text_docs_bow: A list of lists of words from a document
        more_words: An optional list of words to remove along with the stop words

    Returns:
        List of lists of words for each text document without single character words

    """
    stop_words_original = set(stopwords.words('english'))

    if more_words:
        stop_words = stop_words_original.union(more_words)
    else:
        stop_words = stop_words_original

    new_text_docs_bow = [[word for word in doc if word not in stop_words] for doc in text_docs_bow]

    return new_text_docs_bow


def lemmatize(text_docs_bow):
    """Lemmatizes all words in documents

    Args:
        text_docs_bow: A lists of list of words from a document

    Returns:
        List of lists of lemmatized words for each text document

    """
    lemmatizer = WordNetLemmatizer()
    new_text_docs_bow = [[lemmatizer.lemmatize(word) for word in doc] for doc in text_docs_bow]

    return new_text_docs_bow


def stem(text_docs_bow):
    """Stems all words in documents

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        List of lists of stemmed words for each text document

    """
    stemmer = LancasterStemmer()
    new_text_docs_bow = [[stemmer.stem(word) for word in doc] for doc in text_docs_bow]

    return new_text_docs_bow


def bag_of_words_to_docs(text_docs_bow):
    """Converts list of documents in "bag of words" format back into form of document being stored in one string

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        List of strings of text documents

    """
    new_text_docs = [' '.join(doc) for doc in text_docs_bow]

    return new_text_docs


def create_tfidf_matrix(text_docs):
    """Creates a TF_IDF matrix

    Args:
        text_docs: A list of strings of text documents

    Returns:
        Pandas DataFrame of TF-IDF matrix with documents as rows and words as columns

    """
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(text_docs).toarray()
    df = pd.DataFrame(matrix, columns=tfidf.get_feature_names())
    return df


def create_doc_term_matrix(text_docs):
    """Creates a document-term matrix which gives wordcount per document

    Args:
        text_docs: A list of strings of text documents

    Returns:
        Pandas DataFrame of document-term matrix with documents as rows and words as columns

    """
    countvec = CountVectorizer()
    matrix = countvec.fit_transform(text_docs).toarray()
    df = pd.DataFrame(matrix, columns=countvec.get_feature_names())
    return df


def preprocess_texts(text_docs_bow, lem=False, stem=False, custom_pipeline=None):
    """Cleans list of documents by running through customizable text-preprocessing pipeline

    Args:
        text_docs_bow: A list of list of words from a document (also accepts arrays and Pandas series)
        lem: If True, lemmatization becomes part of the pre-processing. Recommended to set as False and run user-created
        lemmatization function if pipeline is customized. Default is False.
        stem: If True, stemming becomes part of the pre-processing. Recommended to set as False and run user-created
        stemming function if pipeline is customized. Default is False.
        custom_pipeline: A custom list of strings and/or function objects which are the function names that
                         text_docs_bow will run through. Default is False, which uses the pipeline:
                         ['tokenize', 'to_lower', 'remove_punct', 'remove_digits', 'remove_single_char_and_spaces',
                         'remove_stopwords']

    Returns:
        List of lists of words for each document which have undergone a pre-processing pipeline

    """
    if type(text_docs_bow) != list:
        text_docs_bow = list(text_docs_bow)

    if not custom_pipeline:
        pipeline = ['tokenize', 'to_lower', 'remove_punct', 'remove_digits', 'remove_single_char_and_spaces',
                    'remove_stopwords']
    else:
        pipeline = custom_pipeline

    if lem:
        pipeline.append('lemmatize')
    if stem:
        pipeline.append('stem')

    for function in pipeline:
        if isinstance(function, str):
            current_method = getattr(mwdata.text.text_preprocessing, function)
            text_docs_bow = current_method(text_docs_bow)
        else:
            text_docs_bow = function(text_docs_bow)

    return text_docs_bow


def create_ngrams(text_docs, n):
    """Generates "n-grams" from the text document

        Args:
            text_docs: A list of text documents in string format
            n: number of items for n-gram sequence

        Returns:
            List n-grams words for text document

    """
    text_docs_n = preprocess_texts(text_docs, custom_pipeline=['tokenize'])
    text_docs_f = [w for doc in text_docs_n for w in doc if w != ' ']
    n_grams_doc = list(ngrams(text_docs_f, n))
    return n_grams_doc
