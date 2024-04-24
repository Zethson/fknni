from importlib.metadata import version

__all__ = ["faiss"]

from .faiss import FaissImputer

__version__ = version("fknni")
