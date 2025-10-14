"""Data models package for RAG application.

This package provides core data structures used throughout the application.
"""

from src.models.document import (
    ChatHistory,
    ChatMessage,
    Chunk,
    Document,
    SearchResult,
)

__all__ = [
    'Document',
    'Chunk',
    'SearchResult',
    'ChatMessage',
    'ChatHistory',
]
