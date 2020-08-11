from IPython import get_ipython
import pyLDAvis
import warnings


def viz_pyLDAvis(model, corpus, dictionary):
    pyLDAvis.enable_notebook()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="pyLDAvis",
            message="Sorting because non-concatenation axis is not aligned.",
        )
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    return vis
