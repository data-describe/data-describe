try:
    import modin.pandas as frame
    _MODIN_INSTALLED = True
except ImportError as e:
    _MODIN_INSTALLED = False