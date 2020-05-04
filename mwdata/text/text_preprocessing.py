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
from nltk import FreqDist
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='gensim')
from gensim.corpora.dictionary import Dictionary

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
try:
    nltk.data.find('stem/wordnet')
except:
    nltk.download('wordnet')
try:
    nltk.data.find('stopwords')
except:
    nltk.download('stopwords')


def tokenize(text_docs):
    """Turns list of documents into "bag of words" format

    Args:
        text_docs: A list of text documents in string format
    
    Returns:
        new_text_docs_bow: List of lists of words for each text document
    """
    new_text_docs_bow = [word_tokenize(doc) for doc in text_docs]
    return new_text_docs_bow


def to_lower(text_docs_bow):
    """Converts all letters in documents ("bag of words" format) to lowercase

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        new_text_docs_bow: List of lists of lowercase words for each text document
    """
    new_text_docs_bow = [[word.lower() for word in doc] for doc in text_docs_bow]
    return new_text_docs_bow


def remove_punct(text_docs_bow, replace_char=" ", remove_all=False):
    """Removes all punctuation from documents

    Args:
        text_docs_bow: A list of lists of words from a document
        replace_char: Character to replace puncutation instances with. Default is space
        remove_all: If True, removes all instances of punctuation from document. Default is False, which only removes
        leading and/or trailing instances

    Returns:
        new_text_docs_bow_final: List of lists of words for each text document without punctuation
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
        new_text_docs_bow: List of lists of words for each text documents without numbers or words containing numbers
    """
    new_text_docs_bow = [[re.sub('\w*\d\w*', '', word) for word in doc] for doc in text_docs_bow]
    return new_text_docs_bow


def remove_single_char_and_spaces(text_docs_bow):
    """Removes all single character words and spaces from documents

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        new_text_docs_bow: List of lists of words for each text document without single character words
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
        more_words: An optional list of words to remove along with the stop words. Default is None

    Returns:
        new_text_docs_bow: List of lists of words for each text document without single character words
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
        new_text_docs_bow: List of lists of lemmatized words for each text document
    """
    lemmatizer = WordNetLemmatizer()
    new_text_docs_bow = [[lemmatizer.lemmatize(word) for word in doc] for doc in text_docs_bow]
    return new_text_docs_bow


def stem(text_docs_bow):
    """Stems all words in documents

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        new_text_docs_bow: List of lists of stemmed words for each text document
    """
    stemmer = LancasterStemmer()
    new_text_docs_bow = [[stemmer.stem(word) for word in doc] for doc in text_docs_bow]
    return new_text_docs_bow


def bag_of_words_to_docs(text_docs_bow):
    """Converts list of documents in "bag of words" format back into form of document being stored in one string

    Args:
        text_docs_bow: A list of lists of words from a document

    Returns:
        new_text_docs: List of strings of text documents
    """
    new_text_docs = [' '.join(doc) for doc in text_docs_bow]
    return new_text_docs


def create_tfidf_matrix(text_docs):
    """Creates a TF_IDF matrix

    Args:
        text_docs: A list of strings of text documents

    Returns:
        matrix_df: Pandas DataFrame of TF-IDF matrix with documents as rows and words as columns
    """
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(text_docs).toarray()
    matrix_df = pd.DataFrame(matrix, columns=tfidf.get_feature_names())
    return matrix_df


def create_doc_term_matrix(text_docs):
    """Creates a document-term matrix which gives wordcount per document

    Args:
        text_docs: A list of strings of text documents

    Returns:
        matrix_df: Pandas DataFrame of document-term matrix with documents as rows and words as columns
    """
    countvec = CountVectorizer()
    matrix = countvec.fit_transform(text_docs).toarray()
    matrix_df = pd.DataFrame(matrix, columns=countvec.get_feature_names())
    return matrix_df


def preprocess_texts(text_docs, lem=False, stem=False, custom_pipeline=None):
    """Cleans list of documents by running through customizable text-preprocessing pipeline

    Args:
        text_docs: A list of strings of text documents (also accepts arrays and Pandas series)
        lem: If True, lemmatization becomes part of the pre-processing. Recommended to set as False and run
        user-created lemmatization function if pipeline is customized. Default is False
        stem: If True, stemming becomes part of the pre-processing. Recommended to set as False and run
        user-created stemming function if pipeline is customized. Default is False
        custom_pipeline: A custom list of strings and/or function objects which are the function names that
        text_docs_bow will run through. Default is None, which uses the pipeline:
                         ['tokenize', 'to_lower', 'remove_punct', 'remove_digits', 'remove_single_char_and_spaces',
                         'remove_stopwords']

    Returns:
        text_docs: List of lists of words for each document which have undergone a pre-processing pipeline
    """
    if type(text_docs) != list:
        text_docs = list(text_docs)

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
            text_docs = current_method(text_docs)
        else:
            text_docs = function(text_docs)

    return text_docs


def ngram_freq(text_docs, n=3, only_n=False):
    """Generates frequency distribution of "n-grams" from all of the text documents

    Args:
        text_docs: A list of text documents in string format
        n: Highest 'n' for n-gram sequence to include. Default is 3
        only_n: If True, will only include n-grams for specified value of 'n'. Default is False, which also includes
        n-grams for all numbers leading up to 'n'

    Returns:
        freq: Dictionary which contains all identified n-grams as keys and their respective counts as their values
    """
    if n < 2:
        raise ValueError("'n' must be a number 2 or greater")

    freq = FreqDist()
    for line in text_docs:
        tokens = preprocess_texts([line], custom_pipeline=[tokenize, remove_stopwords, remove_punct, remove_digits,
                                                           remove_single_char_and_spaces])
        if only_n:
            current_ngrams = list(ngrams(tokens[0], n))
            freq.update(current_ngrams)
        else:
            for num in range(2, n + 1):
                current_ngrams = list(ngrams(tokens[0], num))
                freq.update(current_ngrams)
    return freq


def filter_dictionary(text_docs, no_below=10, no_above=0.2):
    """Filters words that appear less than a certain amount of times in the document and returns a Gensim
    dictionary and corpus

    Args:
        text_docs: A list of list of words from a document, can include n-grams up to 3
        no_below: Keep tokens which are contained in at least no_below documents. Default is 10
        no_above: Keep tokens which are contained in no more than no_above portion of documents (fraction of total
        corpus size). Default is 0.2

    Returns:
        dictionary: Gensim Dictionary encapsulates the mapping between normalized words and their integer ids
        corpus: Bag of Words (BoW) representation of documents (token_id, token_count)
    """
    dictionary = Dictionary(text_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(doc) for doc in text_docs]
    return dictionary, corpus
