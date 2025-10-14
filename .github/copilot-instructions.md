# GitHub Copilot Instructions

This RAG (Retrieval-Augmented Generation) CLI application uses Python + Ollama + LangChain + ChromaDB in a layered architecture. All user-facing text and documentation should be in Japanese.

## Architecture Overview

**Layered Structure**: The application follows strict separation of concerns:
- `src/cli.py` + `src/commands/` - CLI interface using Click + Rich
- `src/rag/` - Core RAG functionality (vector store, embeddings, processing, engine)
- `src/models/` - Shared data structures
- `src/utils/` - Cross-cutting concerns (config, logging, file handling)

**Critical Dependencies**: ChromaDB uses `PersistentClient` (not Client) for embedded operation - no separate server needed. Ollama models: `llama3.2` (generation), `nomic-embed-text` (embeddings).

## Key Development Patterns

### Japanese-First Codebase
- **All docstrings, comments, and CLI help text MUST be in Japanese**
- User-facing error messages and console output in Japanese
- Use Python 3.13+ type hints: `dict[str, Any]` not `Dict[str, Any]`

### CLI Command Structure
Commands follow consistent patterns in `src/commands/`:
- Use `@click.command()` with Japanese help text
- Include `--verbose/-v` option for detailed output
- Rich console for formatted tables and status messages
- Error handling with specific exception types from RAG modules

Example from `src/commands/document.py`:
```python
@click.command('add')
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='詳細情報を表示')
def add_command(file_path: str, verbose: bool):
    """ドキュメントをベクトルストアに追加"""
```

### RAG Module Dependencies
Use dependency injection pattern - pass instances rather than global imports:
```python
# In commands, initialize dependencies
config = get_config()
vector_store = VectorStore(config)
embedding_generator = EmbeddingGenerator(config)
engine = RAGEngine(vector_store, embedding_generator, config)
```

### Configuration Management
`src/utils/config.py` handles environment variables via python-dotenv:
- `OLLAMA_BASE_URL` (default: http://localhost:11434)
- `CHROMA_PERSIST_DIRECTORY` (default: ./chroma_db)
- `CHUNK_SIZE`, `CHUNK_OVERLAP` for text processing

## Essential Commands

**Development setup**:
```bash
uv sync                           # Install dependencies
uv run python src/cli.py --help   # Run CLI directly
uv run rag --help                 # Run as installed package
```

**Ollama prerequisites** (must be running):
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Critical Implementation Details

### ChromaDB Integration
Always use `chromadb.PersistentClient(path="./chroma_db")` - the application runs ChromaDB embedded, not as a service. Collections are auto-created with `get_or_create_collection()`.

### LangChain Usage
Import from specific packages:
```python
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### Command Registration Pattern
In `src/cli.py`, commands are registered with descriptive names:
```python
cli.add_command(add_command, name="add")
cli.add_command(query, name="query")
```

### Error Handling
Each module defines specific exceptions (e.g., `VectorStoreError`, `EmbeddingError`) that commands catch and display with Rich formatting.

## File Processing Workflow

Documents flow: File Input → `DocumentProcessor.load_document()` → Split into chunks → `EmbeddingGenerator.embed_documents()` → Store in ChromaDB via `VectorStore.add_documents()`

Query flow: User question → Embed query → Search similar chunks → Retrieve context → Generate answer with LLM → Return response with sources

## Current Status

The project is **fully implemented** with all CLI commands functional. The main entry point is `src/cli.py` with commands for document management (add, remove, list, clear), querying (query, search, chat), and configuration (init, status, config).

When extending functionality, maintain the existing patterns and ensure all new user-facing text follows the Japanese-first approach.