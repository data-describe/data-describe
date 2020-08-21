from typing import List, Tuple, TYPE_CHECKING
import warnings

from data_describe.compat import _compat, requires, _IN_NOTEBOOK

if TYPE_CHECKING:
    gensim = _compat["gensim"]


@requires("pyLDAvis")
def viz_visualize_topic_summary(
    model: gensim.models.ldamodel.LdaModel,  # type: ignore
    corpus: List[List[Tuple[int, int]]],
    dictionary: gensim.corpora.dictionary.Dictionary,  # type: ignore
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

    _compat["pyLDAvis"].enable_notebook()  # type: ignore
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="pyLDAvis",
            message="Sorting because non-concatenation axis is not aligned.",
        )
        vis = _compat["pyLDAvis"].gensim.prepare(model, corpus, dictionary)  # type: ignore
        return vis
