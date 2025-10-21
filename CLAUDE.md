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
- **MCP** - Model Context Protocol for Claude Desktop integration
- **Pillow** - Image processing for multimodal features

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
# Run main CLI (recommended)
uv run rag [command]

# Or use alternative commands
uv run python src/cli.py [command]
uv run rag-cli [command]

# Example commands
uv run rag init                        # Initialize configuration
uv run rag add ./docs                  # Add documents from directory
uv run rag query "your question"       # Query the knowledge base
uv run rag chat                        # Start interactive chat session

# MCP Server (for Claude Desktop integration)
uv run rag-mcp-server                  # Start MCP server (stdio mode)
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
2. **MCP Server Layer** (`src/mcp/`) - Model Context Protocol server for Claude Desktop integration
3. **Service Layer** (`src/services/`) - Business logic and orchestration
4. **RAG Core** (`src/rag/`) - Core RAG functionality (document processing, embeddings, retrieval)
5. **Data Models** (`src/models/`) - Shared data structures
6. **Utilities** (`src/utils/`) - Cross-cutting concerns (logging, file handling, config)

### Directory Structure
```
src/
├── cli.py                       # CLI entry point with Click
├── commands/                    # Command implementations
│   ├── document.py             # add, remove, list, clear
│   ├── query.py                # query, search, chat
│   └── config.py               # init, status, config
├── mcp/                         # MCP Server for Claude Desktop
│   ├── server.py               # MCP server implementation
│   ├── handlers.py             # Request handlers (tools, resources, prompts)
│   ├── tools.py                # MCP tool definitions
│   └── resources.py            # MCP resource providers
├── services/                    # Business logic layer
│   ├── document_service.py     # Document management orchestration
│   └── file_utils.py           # File operation utilities
├── rag/                         # RAG core functionality
│   ├── vector_store.py         # ChromaDB operations (PersistentClient)
│   ├── embeddings.py           # Text embedding generation via Ollama
│   ├── vision_embeddings.py    # Image embedding generation
│   ├── document_processor.py   # Text file loading and chunking
│   ├── image_processor.py      # Image file processing
│   ├── engine.py               # Text RAG orchestration (retrieve + generate)
│   └── multimodal_engine.py    # Multimodal RAG (text + image)
├── models/                      # Data models
│   └── document.py             # Document, Chunk, SearchResult, ChatMessage
└── utils/                       # Utilities
    └── config.py               # Config management (python-dotenv)
```

### Key Workflows

**Text Document Ingestion Flow:**
```
File Input → Load (document_processor) → Split into Chunks → Generate Embeddings (embeddings)
→ Store in ChromaDB (vector_store)
```

**Image Document Ingestion Flow:**
```
Image File → Load & Process (image_processor) → Generate Vision Embeddings (vision_embeddings)
→ Store with Metadata in ChromaDB (vector_store)
```

**Text Query/Answer Flow:**
```
User Question → Embed Query (embeddings) → Search Similar Chunks (vector_store)
→ Retrieve Context → Generate Answer with LLM (engine) → Return Response
```

**Image Search Flow:**
```
Text Query → Embed Query (embeddings) → Search Similar Images (vector_store)
→ Retrieve Image Results → Return with Metadata
```

**MCP Server Flow:**
```
Claude Desktop → MCP Request (stdio) → Handler (handlers.py) → Service Layer (document_service)
→ RAG Core → Response → Claude Desktop
```

### ChromaDB Usage Pattern
- Uses `PersistentClient` (not Client) for data persistence
- Default storage path: `./chroma_db`
- No separate server process required - runs embedded in Python
- Collections organized by content type:
  - `documents` - Text document chunks
  - `images` - Image embeddings and metadata

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

✅ **All Core Features Complete**
- Text-based RAG system fully implemented and tested
- Multimodal features (image processing and search) implemented
- MCP Server for Claude Desktop integration
- CLI interface with comprehensive commands
- Comprehensive unit and integration test coverage

**Implemented Components:**

*CLI Layer:*
- [src/cli.py](src/cli.py) - Main CLI entry point with Click groups
- [src/commands/document.py](src/commands/document.py) - Document management (add, remove, list, clear)
- [src/commands/query.py](src/commands/query.py) - Query operations (query, search, chat)
- [src/commands/config.py](src/commands/config.py) - Configuration commands (init, status, config)

*MCP Server Layer:*
- [src/mcp/server.py](src/mcp/server.py) - MCP server implementation with stdio transport
- [src/mcp/handlers.py](src/mcp/handlers.py) - Tool, resource, and prompt handlers
- [src/mcp/tools.py](src/mcp/tools.py) - MCP tool definitions (add_document, search, etc.)
- [src/mcp/resources.py](src/mcp/resources.py) - Resource providers (document list)

*Service Layer:*
- [src/services/document_service.py](src/services/document_service.py) - Business logic for document operations
- [src/services/file_utils.py](src/services/file_utils.py) - File utilities and type detection

*RAG Core:*
- [src/rag/vector_store.py](src/rag/vector_store.py) - ChromaDB vector operations
- [src/rag/embeddings.py](src/rag/embeddings.py) - Text embedding generation via Ollama
- [src/rag/vision_embeddings.py](src/rag/vision_embeddings.py) - Image embedding generation
- [src/rag/document_processor.py](src/rag/document_processor.py) - Text document loading and chunking
- [src/rag/image_processor.py](src/rag/image_processor.py) - Image file processing
- [src/rag/engine.py](src/rag/engine.py) - Text RAG orchestration with chat history
- [src/rag/multimodal_engine.py](src/rag/multimodal_engine.py) - Multimodal RAG engine

*Data Models:*
- [src/models/document.py](src/models/document.py) - Document, Chunk, SearchResult, ChatMessage models

*Utilities:*
- [src/utils/config.py](src/utils/config.py) - Configuration management with .env support

**Test Coverage:**
- **Unit tests** for all core modules ([tests/unit/](tests/unit/))
  - Text processing: test_document_processor.py, test_embeddings.py
  - Image processing: test_image_processor.py, test_vision_embeddings.py
  - Engines: test_engine.py, test_multimodal_engine.py
  - Services: test_document_service.py, test_file_utils.py
  - Data models: test_models.py
  - Storage: test_vector_store.py
- **Integration tests** for real-world scenarios ([tests/integration/](tests/integration/))
  - test_ollama_integration.py - Ollama connectivity
  - test_full_rag_flow.py - End-to-end text RAG
  - test_multimodal_rag.py - Multimodal RAG flow
  - test_image_search.py - Image search functionality
  - test_multimodal_search.py - Combined text and image search
- Pytest fixtures for common test setup ([tests/conftest.py](tests/conftest.py))

**Current Branch:** `main`

**Next Steps:**
- Performance optimization and caching
- Additional file format support (DOCX, HTML, etc.)
- Advanced search features (filters, facets)
- Export/import functionality
- Enhanced CLI user experience

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
5. For multimodal features, maintain separation between text and image processing pipelines

### When Working with MCP Server
1. All MCP handlers should delegate business logic to the service layer
2. Keep MCP layer thin - only handle protocol concerns (request/response formatting)
3. Use proper error handling and return meaningful error messages to Claude Desktop
4. Test MCP tools with actual Claude Desktop integration when possible
5. MCP server runs in stdio mode - avoid any print statements that interfere with protocol communication

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
- Use pytest markers for test categorization:
  - `@pytest.mark.unit` - Unit tests (no external dependencies)
  - `@pytest.mark.integration` - Integration tests (requires Ollama)
  - `@pytest.mark.multimodal` - Multimodal feature tests
  - `@pytest.mark.performance` - Performance tests
  - `@pytest.mark.slow` - Long-running tests

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

# Text document collection
collection = client.get_or_create_collection(name="documents")

# Image collection
image_collection = client.get_or_create_collection(name="images")
```

### MCP Tool Implementation
```python
from mcp.types import Tool, TextContent
from src.services.document_service import DocumentService

# Define tool
add_document_tool = Tool(
    name="add_document",
    description="Add a document to the RAG system",
    inputSchema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"}
        },
        "required": ["file_path"]
    }
)

# Implement handler
async def handle_add_document(arguments: dict) -> list[TextContent]:
    service = DocumentService()
    result = service.add_document(arguments["file_path"])
    return [TextContent(type="text", text=f"Added: {result}")]
```

### Service Layer Pattern
```python
# Services orchestrate multiple RAG components
from src.rag.vector_store import VectorStore
from src.rag.document_processor import DocumentProcessor

class DocumentService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.processor = DocumentProcessor()

    def add_document(self, file_path: str) -> dict:
        # Load and process
        chunks = self.processor.load_and_split(file_path)
        # Store in vector DB
        doc_id = self.vector_store.add_chunks(chunks)
        return {"id": doc_id, "chunks": len(chunks)}
```

## Reference Documentation
- Feature specifications: [docs/overview.md](docs/overview.md)
- MCP Server plan: [docs/mcp-server-implementation-plan.md](docs/mcp-server-implementation-plan.md)
- Vector DB migration: [docs/vector-db-migration-plan.md](docs/vector-db-migration-plan.md)

## Key Features

### Text RAG
- TXT, PDF, Markdown file support
- Configurable chunking (size, overlap)
- Semantic search with ChromaDB
- Context-aware answer generation
- Chat mode with conversation history

### Multimodal RAG
- Image file support (JPG, PNG, GIF, BMP, WEBP)
- Vision embeddings for images
- Text-to-image search
- Image metadata storage
- Combined text and image search

### MCP Server Integration
- Claude Desktop integration via Model Context Protocol
- Tools: add_document, search, search_images, list_documents, remove_document
- Resources: Document list with metadata
- stdio-based communication
- Seamless integration with existing RAG core

### CLI Features
- Rich terminal UI with tables and colors
- Interactive chat mode
- Progress indicators for long operations
- Comprehensive error messages
- Configuration management
