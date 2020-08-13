import warnings

from data_describe import compat


def viz_pyLDAvis(model, corpus, dictionary):
    """Displays interactive pyLDAvis visual to understand topic model and documents.

    Args:
        model: LDA topic model
        corpus: Bag of Words (BoW) representation of documents (token_id, token_count)
        dictionary: Gensim Dictionary encapsulates the mapping between normalized words and their integer ids

    Returns:
        A visual to understand topic model and/or documents relating to model
    """
    if compat.get_ipython() is None:
        raise EnvironmentError("Not in Jupyter Notebook")
    else:
        compat.pyLDAvis.enable_notebook()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="pyLDAvis", message="Sorting because non-concatenation axis is not aligned.",)
            vis = compat.pyLDAvis.gensim.prepare(model, corpus, dictionary)
            return vis
