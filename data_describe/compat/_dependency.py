from importlib.util import find_spec
from importlib import import_module
import warnings
from functools import wraps
from types import ModuleType
from typing import Dict, Callable, List


class DependencyManager:
    """Manage optional dependencies for data-describe.

    Examples:
        Optional modules can be accessed as an attribute::

            _compat = DependencyManager({"presidio": None})
            engine = _compat.presidio_analyzer.AnalyzerEngine()
    """

    def __init__(self, imports: Dict[str, Callable]):
        """Initializes the expected optional dependencies for data-describe.

        Args:
            imports (Dict[str, Callable]): A dictionary of module names with optional callables.
                The Callable will be executed when the module is imported, and can be used to
                run additional download or other set up processes that should happen on import.
        """
        self.imports = imports
        self.installed_modules: Dict[str, bool] = {}
        self.modules: Dict[str, ModuleType] = {}
        self.search_install(list(imports.keys()))

    def search_install(self, modules: List[str]):
        """Searches for installed modules and determines if they exist.

        The attribute `installed_modules` maps module names to booleans
        for modules which are found.

        Args:
            modules (List[str]): List of module names
        """
        for module in modules:
            if find_spec(module) is not None:
                self.installed_modules[module] = True
            else:
                self.installed_modules[module] = False

    def check_install(self, module: str) -> bool:
        """Checks to see if a module is installed."""
        return self.installed_modules[module]

    def __getattr__(self, item: str) -> ModuleType:
        """Allows attribute-style access to optional modules.

        Args:
            item (str): The module.

        Returns:
            The module.
        """
        if item in self.installed_modules.keys():
            if self.installed_modules[item]:
                if item in self.modules.keys():
                    return self.modules[item]
                else:
                    try:
                        module = import_module(item)
                        self.modules[item] = module
                        if self.imports[item] is not None:
                            self.imports[item](module)
                        return module
                    except ImportError:
                        raise ImportError(
                            f"Unable to import {item} which is required by this feature."
                        )
        raise AttributeError(
            f"Requested module dependency {item} was not initialized by data-describe"
        )


def nltk_download(module):
    """Downloads NLTK corpora."""
    try:
        module.data.find("tokenizers/punkt")
    except LookupError:
        module.download("punkt")
    try:
        module.data.find("corpora/wordnet")
    except LookupError:
        module.download("wordnet")
    try:
        module.data.find("corpora/stopwords")
    except LookupError:
        module.download("stopwords")


def spacy_download(module):
    """Downloads SpaCy language model."""
    if not module.util.is_package("en_core_web_lg"):
        warnings.warn(
            "Downloading en_core_web_lg model for Spacy. This may take several minutes."
        )
        module.cli.download("en_core_web_lg")


def no_side_import(module):
    """Placeholder for imports that should not do anything additional on import."""
    pass


_compat = DependencyManager(
    {
        "nltk": nltk_download,
        "gensim": no_side_import,
        "pyLDAvis": no_side_import,
        "gcsfs": no_side_import,
        "google.cloud.storage": no_side_import,
        "spacy": spacy_download,
        "modin": no_side_import,
        "hdbscan": no_side_import,
        "presidio_analyzer": no_side_import,
    }
)


def requires(package_name):
    """Marks a method or class that requires an optional dependency."""

    def f(func):
        @wraps(func)
        def g(*args, **kwargs):
            if not _compat.check_install(package_name):
                raise ImportError(f"{package_name} required to use this feature.")
            return func(*args, **kwargs)

        return g

    return f
