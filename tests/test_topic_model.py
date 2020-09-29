import pandas as pd
import numpy as np
import gensim
import matplotlib
import sklearn
import pytest

from data_describe.text.topic_modeling import topic_model


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
            random_state=1,
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
    assert lda_model.model_type == "LDA", "Model type attribute is incorrect"
    assert isinstance(
        lda_model.model, gensim.models.ldamodel.LdaModel
    ), "Model type is incorrect"
    assert lda_model.num_topics == len(
        lda_model._model.get_topics()
    ), "Number of topics attribute is incorrect"
    assert (
        len(lda_model.coherence_values)
        == lda_model.max_topics - lda_model.min_topics + 1
    ), "Length of coherence values list is incorrect"
    assert lda_model.min_topics == 2, "Minimum topics attribute is incorrect"
    assert lda_model.max_topics == 6, "Maximum topics attribute is incorrect"


def test_lsi_attributes(lsi_model):
    assert lsi_model.model_type == "LSI", "Model type attribute is incorrect"
    assert isinstance(
        lsi_model.model, gensim.models.lsimodel.LsiModel
    ), "Model type is incorrect"
    assert lsi_model.num_topics == len(
        lsi_model._model.get_topics()
    ), "Number of topics attribute is incorrect"
    assert (
        len(lsi_model.coherence_values)
        == lsi_model.max_topics - lsi_model.min_topics + 1
    ), "Length of coherence values list attribute is incorrect"
    assert lsi_model.min_topics == 2, "Minimum topics attribute is incorrect"
    assert lsi_model.max_topics == 6, "Maximum topics attribute is incorrect"


def test_svd_attributes(svd_model):
    assert svd_model.model_type == "SVD", "Model type attribute is incorrect"
    assert isinstance(
        svd_model.model, sklearn.decomposition.TruncatedSVD
    ), "Model type is incorrect"
    assert svd_model.num_topics == 5, "Number of topics attribute is incorrect"
    with pytest.raises(AttributeError):
        assert (
            svd_model.coherence_values
        ), "Coherence values attribute incorrectly exists"
    with pytest.raises(AttributeError):
        assert svd_model.min_topics, "Maximum topics attribute incorrectly exist"


def test_nmf_attributes(nmf_model):
    assert nmf_model.model_type == "NMF", "Model type attribute is incorrect"
    assert isinstance(
        nmf_model.model, sklearn.decomposition.NMF
    ), "Model type is incorrect"
    assert nmf_model.num_topics == 3, "Number of topics attribute is incorrect"
    with pytest.raises(AttributeError):
        assert (
            nmf_model.coherence_values
        ), "Coherence values attribute incorrectly exists"
    with pytest.raises(AttributeError):
        assert nmf_model.max_topics, "Maximum topics attribute incorrectly exist"


def test_lda_intermediates(lda_model):
    assert isinstance(
        lda_model.dictionary, gensim.corpora.dictionary.Dictionary
    ), "Output is not of the expected return type of Gensim Dictionary object"
    assert isinstance(
        lda_model.corpus[0], list
    ), "Output does not contain expected type of lists inside of return value"
    with pytest.raises(AttributeError):
        assert lda_model.matrix, "Matrix attribute incorrectly exist"


def test_lsi_intermediates(lsi_model):
    assert isinstance(
        lsi_model.corpus, list
    ), "Output is not of the expected return type of list"
    assert isinstance(
        lsi_model.corpus[0][0], tuple
    ), "Output does not contain expected type of tuple inside of return value inside of the return value"


def test_svd_nmf_intermediates(svd_model, nmf_model):
    assert isinstance(
        svd_model.matrix, pd.DataFrame
    ), "Output is not of the expected return type of Pandas data frame"
    assert isinstance(
        nmf_model.matrix, pd.DataFrame
    ), "Output is not of the expected return type of Pandas data frame"
    assert not svd_model.matrix.equals(
        nmf_model.matrix
    ), "SVD and NMF models incorrectly have the same exact matrix attribute"
    with pytest.raises(AttributeError):
        assert svd_model.corpus, "Corpus attribute incorrectly exist"


def test_visualize_topic_summary(lda_model, svd_model):
    with pytest.raises(OSError):
        lda_model.visualize_topic_summary()
    with pytest.raises(TypeError):
        svd_model.visualize_topic_summary(), "Visualize topic summary function incorrectly runs for non-LDA topic model"


def test_elbow_plot(lsi_model, nmf_model):
    assert isinstance(
        lsi_model.elbow_plot(), matplotlib.axes._subplots.Axes
    ), "Elbow plot is not of the expected return type of a Matplotlib Plot"
    with pytest.raises(TypeError):
        nmf_model.elbow_plot(), "Elbow plot function incorrectly runs for topic model that is not an LDA or LSA/LSI model"


def test_topic_keywords(lda_model, nmf_model):
    lda_keywords_df = lda_model.show(num_topic_words=5)
    assert lda_keywords_df.shape == (
        5,
        2 * lda_model.num_topics,
    ), "Test topic keywords function is the incorrect size"
    assert isinstance(
        lda_keywords_df, pd.DataFrame
    ), "Output is not of the expected return type of Pandas data frame"
    nmf_keywords_df = nmf_model.show()
    assert nmf_keywords_df.shape == (
        10,
        3,
    ), "Test topic keywords function is the incorrect size"
    assert isinstance(
        nmf_keywords_df, pd.DataFrame
    ), "Output is not of the expected return type of Pandas data frame"


def test_topic_top_documents(lsi_model, svd_model, document_data):
    lsi_top_docs_df = lsi_model.top_documents_per_topic(
        text_docs=document_data, num_docs=2
    )
    assert isinstance(
        lsi_top_docs_df, pd.DataFrame
    ), "Output is not of the expected return type of Pandas data frame"
    assert lsi_top_docs_df.shape == (
        2,
        lsi_model.num_topics,
    ), "Top documents per topic function return value is the incorrect size"

    svd_top_docs_df = svd_model.top_documents_per_topic(
        text_docs=document_data, num_docs=2
    )
    svd_top_docs_df_summarized = svd_model.top_documents_per_topic(
        text_docs=document_data, summarize_docs=True, num_docs=2,
    )
    svd_top_docs_df_summarized_words = svd_model.top_documents_per_topic(
        text_docs=document_data, summarize_docs=True, summary_words=15, num_docs=2,
    )
    assert not svd_top_docs_df.equals(
        svd_top_docs_df_summarized
    ), "Output of top documents per topic function for non-summarized documents is incorrectly equal to output of the function with summarized documents"
    assert not svd_top_docs_df_summarized.equals(
        svd_top_docs_df_summarized_words
    ), "Output of top documents per topic function for 10-word-summarized documents is incorrectly equal to output of the function with 15-word-summarized documents"
    assert svd_top_docs_df.shape == (
        2,
        4,
    ), "Top documents per topic function return value is the incorrect size"
