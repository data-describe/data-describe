from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import Phrases
from mwdata import load_data
from mwdata.utilities.contextmanager import _context_manager
from IPython import get_ipython
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis.gensim
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='nltk', message='`formatargspec` is deprecated since Python 3.5')
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='gensim', message='Calling np.sum(generator) is deprecated')
    import nltk
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)


@_context_manager
def topic_modeling(df, col, num_topics=None, elbow=False, lda_kwargs=None, context=None):
    """Function for creating an interactive plot summarizing document topics

    Args:
        df: Pandas data frame or file path
        col: column containing text
        lda_kwargs: key words to be passed into LdaModel
        num_topics: number of topics for LDA model
        elbow: if True, creates an elbow plot for coherence values and number of topics
        context: the context

    Returns:
         vis: pyLDAvis visualization and elbow plot if elbow=True

    """
    if isinstance(df, pd.DataFrame):
        docs = df[col].astype(str).to_numpy()
    elif isinstance(df, str):
        data = load_data(df)
        docs = data[col].astype(str).to_numpy()

    if elbow is True and num_topics is not None:
        raise ValueError('num_topics must be None to use elbow')

    docs = docs_preprocessor(docs)
    docs = extract_ngrams(docs, bigram_min_count=10, trigram_min_count=1)
    gensim_dictionary, documents = filter_dictionary(docs)

    if elbow is True or num_topics is None:
        model, topic_numbers, values = compute_model(dictionary=gensim_dictionary, corpus=documents, texts=docs,
                                                  num_topics=num_topics, lda_kwargs=lda_kwargs, context=context)
    else:
        model = compute_model(dictionary=gensim_dictionary, corpus=documents, texts=docs,
                                                num_topics=num_topics, lda_kwargs=lda_kwargs, context=context)
    if get_ipython() is not None:
        pyLDAvis.enable_notebook()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module='pyLDAvis',
                                    message='Sorting because non-concatenation axis is not aligned.')
            vis = pyLDAvis.gensim.prepare(model, documents, gensim_dictionary)
        if elbow is True:
            fig = plot_elbow(topic_numbers, values, context=context)
            return vis, fig
        else:
            return vis
    else:
        raise EnvironmentError("Not in Jupyter Notebook")


def docs_preprocessor(docs):
    """Cleans text for LDA model

    Args:
        docs: docs in numpy.ndarray format

    Returns:
        docs: list type
    """
    tokenizer = RegexpTokenizer(r'\w+')
    docs = [tokenizer.tokenize(doc.lower()) for doc in docs]
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]
    docs = [[token for token in doc if len(token) > 3] for doc in docs]
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    return docs


def extract_ngrams(docs, bigram_min_count=10, trigram_min_count=1):
    """Extracts bigrams and trigrams

    Args:
        docs: docs, following docs_preprocessor
        bigram_min_count: minimum count of a bigram required
        trigram_min_count: minimum count of a trigram required

    Returns:
        docs: list containing bigrams and trigrams

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=r'For a faster implementation,')
        bigram = Phrases(docs, min_count=bigram_min_count)
        trigram = Phrases(bigram[docs], min_count=trigram_min_count)

        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    docs[idx].append(token)
            for token in trigram[docs[idx]]:
                if '_' in token:
                    docs[idx].append(token)
        return docs


def filter_dictionary(docs, no_below=10, no_above=0.2):
    """Filters words that appear less than 10 times in the document

    Args:
        docs: list containing up to n-gram = 3
        no_below: (int) Keep tokens which are contained in at least no_below documents
        no_above: (float) Keep tokens which are contained in no more than no_above documents (fraction of total corpus size)

    Returns:
        dictionary: Gensim Dictionary encapsulates the mapping between normalized words and their integer ids
        corpus: Bag of Words (BoW) representation of document (token_id, token_count)

    """
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    return dictionary, corpus


def compute_model(dictionary, corpus, texts, num_topics=None, limit=20, start=1, lda_kwargs=None, context=None):
    """Creates LDA model given number of topics or calculates number of topics based on optimized coherence value

    Args:
        dictionary: Gensim dictionary
        corpus: Gensim corpus
        texts: List of input texts
        num_topics: number of topics
        limit: Max number of topics
        start: Start number of topics
        lda_kwargs: Key words to be passed into the LdaModel
        context: the context

    Returns:
        model_list : LdaModel

    """
    coherence_values = []
    model_list = []
    if lda_kwargs is None:
        lda_kwargs = {'chunksize': 500, 'alpha': 'auto', 'eta': 'auto', 'iterations': 100,
                      'passes': 20, 'eval_every': 1}
    r = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if num_topics is None:
            for num_topics in range(start, limit):
                model = LdaModel(corpus=corpus, num_topics=num_topics, **lda_kwargs)
                coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                score = coherence_model.get_coherence()
                if len(coherence_values) == 0:
                    coherence_values.append(score)
                    model_list.append(model)
                    r.append(num_topics)
                elif score > coherence_values[-1]:
                    coherence_values.append(score)
                    model_list.append(model)
                    r.append(num_topics)
                else:
                    break
            return model_list[-1], r, coherence_values

        else:
            model = LdaModel(corpus=corpus, num_topics=num_topics, **lda_kwargs)
            return model


@_context_manager
def plot_elbow(topic_numbers, values, context=None):
    """Creates an elbow plot for coherence values and number of topics

    Args:
        topic_numbers: number of topics
        values: coherence values
        context: the context

    Returns:
        fig : Matplolib lineplot

    """
    plt.figure(figsize=(context.fig_width, context.fig_height))
    fig = sns.lineplot(x=topic_numbers, y=values)
    fig.set_title('Coherence values across topic numbers')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Values')
    plt.show()
    return fig
