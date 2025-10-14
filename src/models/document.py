"""Data models for RAG application.

This module defines the core data structures used throughout the application:
- Document: Represents a source document with metadata
- Chunk: Represents a split text chunk with metadata
- SearchResult: Represents a search result with similarity score
- ChatMessage: Represents a chat message in conversation history
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """Represents a source document with metadata.

    Attributes:
        file_path: Absolute path to the source file
        name: Document name (typically filename)
        content: Full text content of the document
        doc_type: File type/extension (e.g., 'txt', 'pdf', 'md')
        source: Source identifier (typically file path string)
        timestamp: Creation/addition timestamp
        metadata: Additional metadata dictionary
    """
    file_path: Path
    name: str
    content: str
    doc_type: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Returns the character count of the document content."""
        return len(self.content)

    def __post_init__(self):
        """Ensure file_path is a Path object."""
        if not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)


@dataclass
class Chunk:
    """Represents a split text chunk with metadata.

    Attributes:
        content: Text content of the chunk
        chunk_id: Unique identifier for the chunk
        document_id: Reference to parent document
        chunk_index: Index of this chunk in the document (0-based)
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        metadata: Metadata from parent document plus chunk-specific info
    """
    content: str
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Returns the character count of the chunk content."""
        return len(self.content)

    def __post_init__(self):
        """Add chunk-specific metadata."""
        self.metadata.update({
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'size': self.size,
        })


@dataclass
class SearchResult:
    """Represents a search result with similarity score.

    Attributes:
        chunk: The matched text chunk
        score: Similarity score (typically cosine similarity, 0-1 range)
        document_name: Name of the source document
        document_source: Source path/identifier of the document
        rank: Ranking position in search results (1-based)
        metadata: Additional metadata from the chunk
    """
    chunk: Chunk
    score: float
    document_name: str
    document_source: str
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure score is within valid range."""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")


@dataclass
class ChatMessage:
    """Represents a chat message in conversation history.

    Attributes:
        role: Message role ('user', 'assistant', or 'system')
        content: Message content text
        timestamp: Message timestamp
        metadata: Additional metadata (e.g., model used, tokens, context)
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate role."""
        valid_roles = {'user', 'assistant', 'system'}
        if self.role not in valid_roles:
            raise ValueError(
                f"Role must be one of {valid_roles}, got '{self.role}'"
            )

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for LLM APIs.

        Returns:
            Dictionary with 'role' and 'content' keys
        """
        return {
            'role': self.role,
            'content': self.content,
        }


@dataclass
class ChatHistory:
    """Manages conversation history for chat mode.

    Attributes:
        messages: List of chat messages
        max_messages: Maximum number of messages to keep (None = unlimited)
    """
    messages: list[ChatMessage] = field(default_factory=list)
    max_messages: int | None = None

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a new message to history.

        Args:
            role: Message role ('user', 'assistant', or 'system')
            content: Message content
            metadata: Optional metadata dictionary
        """
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)

        # Trim history if max_messages is set
        if self.max_messages and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def to_dicts(self) -> list[dict[str, str]]:
        """Convert all messages to dictionary format for LLM APIs.

        Returns:
            List of dictionaries with 'role' and 'content' keys
        """
        return [msg.to_dict() for msg in self.messages]

    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages.clear()

    def __len__(self) -> int:
        """Return number of messages in history."""
        return len(self.messages)
