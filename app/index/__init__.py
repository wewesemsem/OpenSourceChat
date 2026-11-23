"""Keyword index over allowlisted folders.

Retrieval is ordinary code. This milestone does not give the model a tool.
"""

from app.index.scan import IndexResult, index_folder
from app.index.search import Hit, search_chunks

__all__ = [
    "Hit",
    "IndexResult",
    "index_folder",
    "search_chunks",
]
