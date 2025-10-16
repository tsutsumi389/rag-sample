# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) CLI application using Python + Ollama + LangChain + ChromaDB. This system stores local documents in a vector database and generates answers to natural language questions by searching and retrieving relevant information.

## Tech Stack

- **Python 3.13+** with uv package manager
- **Ollama** - Local LLM execution (gpt-oss for generation, nomic-embed-text for embeddings)
- **LangChain** - LLM application framework (langchain, langchain-community, langchain-ollama)
- **ChromaDB** - Vector database (runs embedded, no separate server needed)
- **Click** - CLI framework
- **Rich** - Terminal UI formatting

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Activate virtual environment (if needed)
source .venv/bin/activate  # macOS/Linux
```

### Running the Application
```bash
# Run main CLI
uv run python src/cli.py [command]

# Or use as installed package
uv run rag-cli [command]

# Example commands
uv run rag-cli init                    # Initialize configuration
uv run rag-cli add ./docs              # Add documents from directory
uv run rag-cli query "your question"   # Query the knowledge base
uv run rag-cli chat                    # Start interactive chat session
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest tests/unit/

# Run integration tests only
uv run pytest tests/integration/

# Run with coverage report
uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/test_engine.py -v
```

### Ollama Prerequisites
The application requires Ollama to be running locally with these models:
```bash
ollama pull gpt-oss           # LLM for answer generation
ollama pull nomic-embed-text  # Embedding model
```

## Architecture

### Core Design Pattern
The application follows a **layered architecture** separating concerns:

1. **CLI Layer** (`src/cli.py`, `src/commands/`) - User interface and command handling
2. **RAG Core** (`src/rag/`) - Business logic for document processing and retrieval
3. **Data Models** (`src/models/`) - Shared data structures
4. **Utilities** (`src/utils/`) - Cross-cutting concerns (logging, file handling, config)

### Directory Structure
```
src/
├── cli.py                    # CLI entry point with Click
├── commands/                 # Command implementations
│   ├── document.py          # add, remove, list, clear
│   ├── query.py             # query, search, chat
│   └── config.py            # init, status, config
├── rag/                     # RAG core functionality
│   ├── vector_store.py      # ChromaDB operations (PersistentClient)
│   ├── embeddings.py        # Embedding generation via Ollama
│   ├── document_processor.py # File loading and chunking
│   └── engine.py            # RAG orchestration (retrieve + generate)
├── models/                  # Data models
│   └── document.py          # Document, Chunk, SearchResult, ChatMessage
└── utils/                   # Utilities
    ├── config.py            # Config management (python-dotenv)
    ├── file_handler.py      # File operations
    └── logger.py            # Logging setup
```

### Key Workflows

**Document Ingestion Flow:**
```
File Input → Load (document_processor) → Split into Chunks → Generate Embeddings (embeddings)
→ Store in ChromaDB (vector_store)
```

**Query/Answer Flow:**
```
User Question → Embed Query (embeddings) → Search Similar Chunks (vector_store)
→ Retrieve Context → Generate Answer with LLM (engine) → Return Response
```

### ChromaDB Usage Pattern
- Uses `PersistentClient` (not Client) for data persistence
- Default storage path: `./chroma_db`
- No separate server process required - runs embedded in Python
- Collections organized by document type or project

### Configuration Management
Environment variables (via `.env` file):
- `OLLAMA_BASE_URL` - Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_LLM_MODEL` - Model for text generation (default: gpt-oss)
- `OLLAMA_EMBEDDING_MODEL` - Model for embeddings (default: nomic-embed-text)
- `CHROMA_PERSIST_DIRECTORY` - ChromaDB storage path (default: ./chroma_db)
- `CHUNK_SIZE` - Text chunk size (default: 1000)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 200)
- `LOG_LEVEL` - Logging level (default: INFO)

## Implementation Status

✅ **Phase 1-3: Core Implementation Complete**
- All core RAG modules are implemented and tested
- CLI interface with document, query, and config commands
- Comprehensive unit and integration test coverage

**Implemented Components:**

*CLI Layer:*
- [src/cli.py](src/cli.py) - Main CLI entry point with Click groups
- [src/commands/document.py](src/commands/document.py) - Document management (add, remove, list, clear)
- [src/commands/query.py](src/commands/query.py) - Query operations (query, search, chat)
- [src/commands/config.py](src/commands/config.py) - Configuration commands (init, status, config)

*RAG Core:*
- [src/rag/vector_store.py](src/rag/vector_store.py) - ChromaDB vector operations
- [src/rag/embeddings.py](src/rag/embeddings.py) - Ollama embedding generation
- [src/rag/document_processor.py](src/rag/document_processor.py) - Document loading and chunking
- [src/rag/engine.py](src/rag/engine.py) - RAG orchestration with chat history support

*Data Models:*
- [src/models/document.py](src/models/document.py) - Document, Chunk, SearchResult, ChatMessage models

*Utilities:*
- [src/utils/config.py](src/utils/config.py) - Configuration management with .env support

**Test Coverage:**
- Unit tests for all core modules ([tests/unit/](tests/unit/))
- Ollama integration tests ([tests/integration/test_ollama_integration.py](tests/integration/test_ollama_integration.py))
- Full RAG flow end-to-end tests ([tests/integration/test_full_rag_flow.py](tests/integration/test_full_rag_flow.py))
- Pytest fixtures for common test setup ([tests/conftest.py](tests/conftest.py))

**Current Branch:** `feature/unit_test` (merged to main)

**Next Steps:**
- Performance optimization and production hardening
- Additional CLI features (export, import, batch operations)
- Documentation and user guides

## Development Guidelines

### Code Style and Documentation
- **All comments and docstrings must be written in Japanese**
- Use Python 3.13+ type hints (e.g., `dict[str, Any]` instead of `Dict[str, Any]`)
- Follow Google-style docstrings format with Japanese text
- Module docstrings should describe the purpose and main components
- Class/function docstrings should include Japanese descriptions of attributes, parameters, and return values

### When Adding New Commands
1. Implement command logic in appropriate `src/commands/*.py` file
2. Register command in `src/cli.py` Click group
3. Use Rich for formatted output (tables, progress bars, etc.)
4. Handle errors gracefully with informative messages

### When Modifying RAG Logic
1. Keep concerns separated: document processing, embedding, storage, retrieval should be independent modules
2. Use dependency injection pattern - pass dependencies (vector store, embeddings) rather than importing globally
3. Add proper type hints using Python 3.13+ syntax
4. Handle Ollama connection failures gracefully

### Data Models
Use `dataclasses` or `pydantic` for data models. Include metadata for:
- Document: file path, name, type, timestamp, source
- Chunk: text content, metadata, document reference
- SearchResult: chunk, similarity score, source document

### Testing Guidelines
- Write unit tests for all core functionality using pytest
- Use mocks (`unittest.mock`) to isolate components from external dependencies
- Integration tests should verify actual Ollama and ChromaDB interactions
- Use fixtures in [tests/conftest.py](tests/conftest.py) for common test setup
- Test error handling and edge cases (empty inputs, connection failures, etc.)
- Maintain high test coverage (aim for >80%)
- Run tests before committing: `uv run pytest`

## Common Patterns

### LangChain Integration
```python
# Use langchain-ollama for Ollama integration
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Use langchain text splitters for chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### ChromaDB Operations
```python
import chromadb

# Always use PersistentClient for data persistence
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")
```

## Reference Documentation
- Implementation plan: `docs/implementation-plan.md`
- Feature specifications: `docs/overview.md`
- Architecture details documented in overview.md lines 154-181
