from importlib.metadata import version

__version__ = version("fknni")

from .knn import FastKNNImputer

__all__ = ["FastKNNImputer", "FaissImputer"]
