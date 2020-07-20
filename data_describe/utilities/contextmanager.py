import copy
from functools import wraps


class ContextManager:
    """Manages the context and controls figure sizes."""

    def __init__(self):
        """Default plot values."""
        self.fig_width = 11
        self.fig_height = 9
        self.viz_size = 700


def _context_manager(func):
    @wraps(func)
    def f(*args, **kwargs):
        global CONTEXT_MGR
        kwargs["context"] = copy.copy(CONTEXT_MGR)
        return func(*args, **kwargs)

    return f


CONTEXT_MGR = ContextManager()
