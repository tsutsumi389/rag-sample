# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) CLI application using Python + Ollama + LangChain + ChromaDB. This system stores local documents in a vector database and generates answers to natural language questions by searching and retrieving relevant information.

## Tech Stack

- **Python 3.13+** with uv package manager
- **Ollama** - Local LLM execution (llama3.2 for generation, nomic-embed-text for embeddings)
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
# Run main CLI (when implemented)
uv run python src/cli.py [command]

# Or use as installed package
uv run rag-cli [command]
```

### Ollama Prerequisites
The application requires Ollama to be running locally with these models:
```bash
ollama pull llama3.2          # LLM for answer generation
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
- `OLLAMA_LLM_MODEL` - Model for text generation (default: llama3.2)
- `OLLAMA_EMBEDDING_MODEL` - Model for embeddings (default: nomic-embed-text)
- `CHROMA_PERSIST_DIRECTORY` - ChromaDB storage path (default: ./chroma_db)
- `CHUNK_SIZE` - Text chunk size (default: 1000)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 200)
- `LOG_LEVEL` - Logging level (default: INFO)

## Implementation Status

The project structure is currently planned but not yet implemented. See `docs/implementation-plan.md` for the full implementation roadmap with 11 tasks across 4 phases.

Current state: Empty `main.py` placeholder exists. All `src/` modules need to be created.

## Development Guidelines

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
