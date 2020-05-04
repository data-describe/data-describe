import mwdata
import numpy as np
import mwdata.text.topic_model as tm
import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn.datasets import fetch_20newsgroups


@pytest.fixture
def data_loader():
    newsgroups_train = fetch_20newsgroups(subset='train', categories=['alt.atheism'])
    df = pd.DataFrame(newsgroups_train['data']).sample(n=50, replace=True, random_state=1)
    return df


@pytest.fixture
def data_loader_numpy(data_loader):
    return data_loader[0].to_numpy()


@pytest.fixture
def docs_preprocessor(data_loader_numpy):
    docs = tm.docs_preprocessor(data_loader_numpy)
    return docs


@pytest.fixture
def extract_ngrams(docs_preprocessor):
    docs = tm.extract_ngrams(docs_preprocessor)
    return docs


@pytest.fixture
def filter_dictionary(extract_ngrams):
    gensim_dictionary, corpus = tm.filter_dictionary(extract_ngrams, no_below=2)
    return gensim_dictionary, corpus


def test_docs_preprocessor(docs_preprocessor):
    assert isinstance(docs_preprocessor, list)
    assert isinstance(docs_preprocessor[0], list)
    assert isinstance(docs_preprocessor[0][1], str)
    assert docs_preprocessor[0][1].islower()


def test_extract_ngrams(extract_ngrams):
    assert isinstance(extract_ngrams, list)
    assert isinstance(extract_ngrams[0], list)
    assert isinstance(extract_ngrams[0][1], str)
    assert extract_ngrams[0][1].islower()


def test_filter_dictionary(filter_dictionary):
    corpus = filter_dictionary[1]
    assert isinstance(corpus, list)
    assert isinstance(corpus[0], list)
    assert isinstance(corpus[0][0][0], int)
    assert isinstance(corpus[0][0][1], int)


def test_plot_elbow():
    fig = tm.plot_elbow([1, 2, 3], [4, 5, 6])
    assert isinstance(fig, matplotlib.artist.Artist)


def test_topic_modeling(data_loader):
    with pytest.raises(EnvironmentError):
        x, y = tm.topic_modeling(data_loader, col=0, num_topics=None, elbow=True)
        assert y is not None
        assert x is not None
        x, y = tm.topic_modeling(data_loader, col=0, num_topics=5, elbow=False)
        assert y is not None
        assert x is not None


def test_compute_model(monkeypatch, filter_dictionary, extract_ngrams):
    class MockLDAModel:
        def __init__(self, corpus=None, num_topics=100, id2word=None,
                     distributed=False, chunksize=2000, passes=1, update_every=1,
                     alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
                     iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
                     random_state=None, ns_conf=None, minimum_phi_value=0.01,
                     per_word_topics=False, callbacks=None, dtype=np.float32):
            self.model = "LDA"  # Do nothing

    class MockCoherenceModel:
        def __init__(self, model=None, topics=None, texts=None, corpus=None, dictionary=None,
                 window_size=None, keyed_vectors=None, coherence='c_v', topn=20, processes=-1):
            self.model = "Coherence"  # Do nothing

        def get_coherence(self):
            return 1

    monkeypatch.setattr('mwdata.text.topic_model.CoherenceModel', MockCoherenceModel)
    monkeypatch.setattr('mwdata.text.topic_model.LdaModel', MockLDAModel)

    model = tm.compute_model(dictionary=filter_dictionary[0], corpus=filter_dictionary[1], texts=extract_ngrams,
                             num_topics=2, limit=3, start=2, lda_kwargs=None)
    assert model is not None

    model = tm.compute_model(dictionary=filter_dictionary[0], corpus=filter_dictionary[1], texts=extract_ngrams,
                             num_topics=None, limit=5, start=1, lda_kwargs=None)
    assert model is not None


def test_topic_modeling_df(monkeypatch):
    def mock_docs_processor(docs):
        return docs

    def mock_extract_ngrams(docs, bigram_min_count=10, trigram_min_count=1):
        return docs

    def mock_filter_dictionary(docs, no_below=10, no_above=0.2):
        return docs, docs

    def mock_compute_model(dictionary, corpus, texts, num_topics=None, elbow=None, limit=3, start=2, lda_kwargs=None):
        if num_topics is None:
            return dictionary, dictionary, dictionary
        else:
            return dictionary

    monkeypatch.setattr(mwdata.text.topic_model, "docs_preprocessor", mock_docs_processor)
    monkeypatch.setattr(mwdata.text.topic_model, "extract_ngrams", mock_extract_ngrams)
    monkeypatch.setattr(mwdata.text.topic_model, "filter_dictionary", mock_filter_dictionary)
    monkeypatch.setattr(mwdata.text.topic_model, "compute_model", mock_compute_model)


def test_raise_error(data_loader):
    with pytest.raises(ValueError):
        tm.topic_modeling(data_loader, col=0, num_topics=5, elbow=True)
    with pytest.raises(FileNotFoundError):
        tm.topic_modeling('examplefile', col=0, num_topics=5, elbow=True)
