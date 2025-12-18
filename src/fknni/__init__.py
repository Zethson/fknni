from importlib.metadata import version

__version__ = version("fknni")

from .faiss import FastKNNImputer

__all__ = ["FastKNNImputer", "FastKNNImputer"]
