from importlib.metadata import version

__version__ = version("fknni")

from .knn import FastKNNImputer, FaissImputer

__all__ = ["FastKNNImputer", "FaissImputer"]
