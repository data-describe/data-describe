import pandas as pd
import numpy as np
import gensim
import matplotlib
import sklearn
import pytest

from data_describe.text.topic_model import topic_model


@pytest.fixture(autouse=True)
def skip_models(monkeypatch):
    class MockCoherenceModel:
        def __init__(
            self,
            model=None,
            topics=None,
            texts=None,
            corpus=None,
            dictionary=None,
            window_size=None,
            keyed_vectors=None,
            coherence="c_v",
            topn=20,
            processes=-1,
        ):
            pass

        def get_coherence(self):
            return 1

    class MockLdaModel:
        def __init__(
            self,
            corpus=None,
            num_topics=100,
            id2word=None,
            distributed=False,
            chunksize=2000,
            passes=1,
            update_every=1,
            alpha="symmetric",
            eta=None,
            decay=0.5,
            offset=1.0,
            eval_every=10,
            iterations=50,
            gamma_threshold=0.001,
            minimum_probability=0.01,
            random_state=None,
            ns_conf=None,
            minimum_phi_value=0.01,
            per_word_topics=False,
            callbacks=None,
            dtype=np.float32,
        ):
            pass

    class MockLsiModel:
        def __init__(
            self,
            corpus=None,
            num_topics=200,
            id2word=None,
            chunksize=20000,
            decay=1.0,
            distributed=False,
            onepass=True,
            power_iters=2,
            extra_samples=100,
            dtype=np.float64,
        ):
            pass

        def fit(self):
            pass

    monkeypatch.setattr(
        "gensim.models.coherencemodel.CoherenceModel", MockCoherenceModel
    )


@pytest.fixture
def lda_model(document_data):
    lda_model = topic_model(document_data, max_topics=6, no_below=1, no_above=0.8)
    return lda_model


@pytest.fixture
def lsi_model(document_data):
    lsi_model = topic_model(
        document_data, model_type="LSI", max_topics=6, no_below=1, no_above=0.8
    )
    return lsi_model


@pytest.fixture
def svd_model(document_data):
    svd_model = topic_model(
        document_data, model_type="SVD", num_topics=5, no_below=1, no_above=0.8
    )
    return svd_model


@pytest.fixture
def nmf_model(document_data):
    nmf_model = topic_model(
        document_data, model_type="NMF", tfidf=False, no_below=1, no_above=0.8
    )
    return nmf_model


def test_unknown_type():
    with pytest.raises(ValueError):
        topic_model("wrong input")


def test_lda_attributes(lda_model):
    assert lda_model.model_type == "LDA"
    assert isinstance(lda_model.model, gensim.models.ldamodel.LdaModel)
    assert lda_model.num_topics == len(lda_model._model.get_topics())
    assert (
        len(lda_model.coherence_values)
        == lda_model.max_topics - lda_model.min_topics + 1
    )
    assert lda_model.min_topics == 2
    assert lda_model.max_topics == 6


def test_lsi_attributes(lsi_model):
    assert lsi_model.model_type == "LSI"
    assert isinstance(lsi_model.model, gensim.models.lsimodel.LsiModel)
    assert lsi_model.num_topics == len(lsi_model._model.get_topics())
    assert (
        len(lsi_model.coherence_values)
        == lsi_model.max_topics - lsi_model.min_topics + 1
    )
    assert lsi_model.min_topics == 2
    assert lsi_model.max_topics == 6


def test_svd_attributes(svd_model):
    assert svd_model.model_type == "SVD"
    assert isinstance(svd_model.model, sklearn.decomposition.TruncatedSVD)
    assert svd_model.num_topics == 5
    with pytest.raises(AttributeError):
        assert svd_model.coherence_values
    with pytest.raises(AttributeError):
        assert svd_model.min_topics


def test_nmf_attributes(nmf_model):
    assert nmf_model.model_type == "NMF"
    assert isinstance(nmf_model.model, sklearn.decomposition.NMF)
    assert nmf_model.num_topics == 3
    with pytest.raises(AttributeError):
        assert nmf_model.coherence_values
    with pytest.raises(AttributeError):
        assert nmf_model.max_topics


def test_lda_intermediates(lda_model):
    assert isinstance(lda_model.dictionary, gensim.corpora.dictionary.Dictionary)
    assert isinstance(lda_model.corpus[0], list)
    with pytest.raises(AttributeError):
        assert lda_model.matrix


def test_lsi_intermediates(lsi_model):
    assert isinstance(lsi_model.corpus, list)
    assert isinstance(lsi_model.corpus[0][0], tuple)


def test_svd_nmf_intermediates(svd_model, nmf_model):
    assert isinstance(svd_model.matrix, pd.DataFrame)
    assert isinstance(nmf_model.matrix, pd.DataFrame)
    assert not svd_model.matrix.equals(nmf_model.matrix)
    with pytest.raises(AttributeError):
        assert svd_model.corpus


def test_visualize_topic_summary(lda_model, svd_model):
    with pytest.raises(OSError):
        lda_model.visualize_topic_summary()
    with pytest.raises(TypeError):
        svd_model.visualize_topic_summary()


def test_elbow_plot(lsi_model, nmf_model):
    assert isinstance(lsi_model.elbow_plot(), matplotlib.axes._subplots.Axes)
    with pytest.raises(TypeError):
        nmf_model.elbow_plot()


def test_topic_keywords(lda_model, nmf_model):
    lda_keywords_df = lda_model.show(num_topic_words=5)
    assert lda_keywords_df.shape == (5, 2 * lda_model.num_topics)
    assert isinstance(lda_keywords_df, pd.DataFrame)
    nmf_keywords_df = nmf_model.show()
    assert nmf_keywords_df.shape == (10, 3)
    assert isinstance(nmf_keywords_df, pd.DataFrame)


def test_topic_top_documents(lsi_model, svd_model, document_data):
    lsi_top_docs_df = lsi_model.top_documents_per_topic(
        text_docs=document_data, num_docs=2
    )
    assert isinstance(lsi_top_docs_df, pd.DataFrame)
    assert lsi_top_docs_df.shape == (2, lsi_model.num_topics)

    svd_top_docs_df = svd_model.top_documents_per_topic(
        text_docs=document_data, num_docs=2
    )
    svd_top_docs_df_summarized = svd_model.top_documents_per_topic(
        text_docs=document_data, summarize_docs=True, num_docs=2,
    )
    svd_top_docs_df_summarized_words = svd_model.top_documents_per_topic(
        text_docs=document_data, summarize_docs=True, summary_words=15, num_docs=2,
    )
    assert not svd_top_docs_df.equals(svd_top_docs_df_summarized)
    assert not svd_top_docs_df_summarized.equals(svd_top_docs_df_summarized_words)
    assert svd_top_docs_df.shape == (2, 4)
