from typing import List, Tuple
import warnings

from data_describe.compat import _compat, _IN_NOTEBOOK


def viz_visualize_topic_summary(
    model: _compat["gensim"].models.ldamodel.LdaModel,
    corpus: List[List[Tuple[int, int]]],
    dictionary: _compat["gensim"].corpora.dictionary.Dictionary,
):
    """Displays interactive pyLDAvis visual to understand topic model and documents.

    Args:
        model: LDA topic model
        corpus: Bag of Words (BoW) representation of documents (token_id, token_count)
        dictionary: Gensim Dictionary encapsulates the mapping between normalized words and their integer ids

    Returns:
        A visual to understand topic model and/or documents relating to model
    """
    if not _IN_NOTEBOOK:
        raise EnvironmentError("Not in Jupyter Notebook")

    _compat["pyLDAvis"].enable_notebook()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="pyLDAvis",
            message="Sorting because non-concatenation axis is not aligned.",
        )
        vis = _compat["pyLDAvis"].gensim.prepare(model, corpus, dictionary)
        return vis
