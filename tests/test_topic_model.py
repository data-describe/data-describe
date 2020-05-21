import warnings

import pandas as pd
import gensim
import matplotlib
import sklearn
import pytest

from mwdata.text.topic_model import TopicModel

warnings.filterwarnings("ignore", category=UserWarning, module="gensim")


@pytest.fixture
def load_data():
    topical_docs = [
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
        "Unicorns are large, horned, bipedal mammals with short heads, tails, and ears. Unicorns are most commonly seen in countries of Central America, South America, and South Africa. They are among the oldest creatures on this planet and are believed to have lived on prehistoric Earth between 5 and 3 billion years ago."
        "Scientists estimate that the earliest signs of unicorn behavior show up in Central America at least 60 million years ago, with the earliest known example found in the Andean highlands of Peru. The presence of a unicorn behavior in a region where it is otherwise highly uncommon has allowed scientists to study how animals communicate, adapt and survive."
        "In the early 1990s, three scientists had been studying the behavior of the largest of many species of unicorns, the Ganesha unicorn. They found that the animals communicate only in a rudimentary form and were very silent in the presence of humans, dogs or other domestic animals.",
        '"The herd\'s incredible success is exciting because it demonstrates the tremendous potential for using animals for communication among people," says Professor Jo-Ann Faubert of the University of New South Wales, who has been working in the area to study the animals. The scientists believe there have been no other animals to communicate with, or even speak their language, at this stage of their evolution.'
        'What made this animal even more amazing was that it appears to have used a combination of communication techniques in order to successfully avoid its own hunters. Using a simple horn, the unicorn could produce a low-pitched whistle with which it could warn others of impending danger. "That could be the first time we have seen the use of a language in animals that use it to defend themselves," explains Professor Faubert. "It means if this particular unicorn can communicate with other animals"'
        '"I was amazed by the incredible range of characters," the researcher, Peter Henningsen a professor at the University of Aarhus, told The Local.'
        '"he fact that the animals are able to communicate seems extraordinary," he explained.The creatures were living peacefully with their herds of goats – which it seems they often slept in – and are said by the researchers to have been using special dialects, known as pidgin.',
        '"It\'s certainly an extraordinary case," said Henningsen.'
        "The discovery was made by Dr.Stefan R.Käppi, and colleagues from the Centre for the Ecology of Birds, Aarhus University in Denmark, with the help of an aerial drone equipped with a microphone to identify the unique language."
        '"It is quite possible that unicorns use the same language as all other mammals," he said.',
        "Today, scientists confirmed the worst possible outcome: the massive asteroid will collide with Earth in 2027 and cause widespread destruction on a scale of the most devastating natural disaster in our country's history.That's because one of the asteroid's two main fragments "
        "has an extremely hard core made of nickel, diamond and iron — which will be very difficult to remove without a robotic rover to do the work, NASA officials said at a press briefing Thursday. But some of the larger fragments contain other metals, such as nickel and cobalt, which"
        " can be easily broken apart by a rover and then blasted apart by a powerful burst of energy.The first asteroid to hit the Earth is called Chicxulub and will be about 4.4 miles (7 kilometers) in diameter, NASA officials said. The second asteroid is known as 2012 DA14 and will be about 2.7 miles (4.3 km) in diameter."
        "The two large chunks of the asteroid will probably crash into each other when they collide within 1.5 million years, NASA officials said.",
    ]
    return topical_docs


@pytest.fixture
def lda_model(load_data):
    lda_model = TopicModel()
    lda_model.fit(load_data, max_topics=6, no_below=1, no_above=0.8)
    return lda_model


@pytest.fixture
def lsi_model(load_data):
    lsi_model = TopicModel(model_type="LSI")
    lsi_model.fit(load_data, max_topics=6, no_below=1, no_above=0.8)
    return lsi_model


@pytest.fixture
def svd_model(load_data):
    svd_model = TopicModel(model_type="SVD", num_topics=5)
    svd_model.fit(load_data, no_below=1, no_above=0.8)
    return svd_model


@pytest.fixture
def nmf_model(load_data):
    nmf_model = TopicModel(model_type="NMF")
    nmf_model.fit(load_data, tfidf=False, no_below=1, no_above=0.8)
    return nmf_model


def test_unknown_type():
    with pytest.raises(ValueError):
        TopicModel("wrong input")


def test_lda_attributes(lda_model):
    assert lda_model.model_type == "LDA"
    assert isinstance(lda_model.model, gensim.models.ldamodel.LdaModel)
    assert (
        lda_model.num_topics
        == lda_model.coherence_values.index(max(lda_model.coherence_values))
        + lda_model.min_topics
    )
    assert (
        len(lda_model.coherence_values)
        == lda_model.max_topics - lda_model.min_topics + 1
    )
    assert lda_model.min_topics == 2
    assert lda_model.max_topics == 6


def test_lsi_attributes(lsi_model):
    assert lsi_model.model_type == "LSI"
    assert isinstance(lsi_model.model, gensim.models.lsimodel.LsiModel)
    assert (
        lsi_model.num_topics
        == lsi_model.coherence_values.index(max(lsi_model.coherence_values))
        + lsi_model.min_topics
    )
    assert (
        len(lsi_model.coherence_values)
        == lsi_model.max_topics - lsi_model.min_topics + 1
    )
    assert lsi_model.min_topics == 2
    assert lsi_model.max_topics == 6


def test_svd_attributes(svd_model):
    assert svd_model.model_type == "SVD"
    assert isinstance(svd_model.model, sklearn.decomposition.truncated_svd.TruncatedSVD)
    assert svd_model.num_topics == 5
    with pytest.raises(AttributeError):
        assert svd_model.coherence_values
    with pytest.raises(AttributeError):
        assert svd_model.min_topics


def test_nmf_attributes(nmf_model):
    assert nmf_model.model_type == "NMF"
    assert isinstance(nmf_model.model, sklearn.decomposition.nmf.NMF)
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


def test_pyldavis(lda_model, svd_model):
    with pytest.raises(OSError):
        lda_model.show()
    with pytest.raises(TypeError):
        svd_model.show()


def test_elbow_plot(lsi_model, nmf_model):
    assert isinstance(lsi_model.show("elbow"), matplotlib.axes._subplots.Axes)
    with pytest.raises(TypeError):
        nmf_model.show("elbow")


def test_topic_keywords(lda_model, nmf_model):
    lda_keywords_df = lda_model.show(
        "top_words_per_topic", viz_kwargs={"num_topic_words": 5}
    )
    assert lda_keywords_df.shape == (5, 2 * lda_model.num_topics)
    assert isinstance(lda_keywords_df, pd.DataFrame)
    nmf_keywords_df = nmf_model.show("top_words_per_topic")
    assert nmf_keywords_df.shape == (10, 3)
    assert isinstance(nmf_keywords_df, pd.DataFrame)


def test_topic_top_documents(lsi_model, svd_model, load_data):
    lsi_top_docs_df = lsi_model.show(
        "top_documents_per_topic", text_docs=load_data, viz_kwargs={"num_docs": 2}
    )
    assert isinstance(lsi_top_docs_df, pd.DataFrame)
    assert lsi_top_docs_df.shape == (2, lsi_model.num_topics)

    svd_top_docs_df = svd_model.show(
        "top_documents_per_topic", text_docs=load_data, viz_kwargs={"num_docs": 2}
    )
    svd_top_docs_df_summarized = svd_model.show(
        "top_documents_per_topic",
        text_docs=load_data,
        viz_kwargs={"summarize_docs": True, "num_docs": 2},
    )
    svd_top_docs_df_summarized_words = svd_model.show(
        "top_documents_per_topic",
        text_docs=load_data,
        viz_kwargs={"summarize_docs": True, "summary_words": 15, "num_docs": 2},
    )
    assert not svd_top_docs_df.equals(svd_top_docs_df_summarized)
    assert not svd_top_docs_df_summarized.equals(svd_top_docs_df_summarized_words)
    assert svd_top_docs_df.shape == (2, 4)
