# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a full-stack RAG (Retrieval-Augmented Generation) chatbot for querying course materials. The system uses Claude with tool calling to perform semantic searches over course content stored in ChromaDB.

## Development Commands

### Running the Application

```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Dependency Management

```bash
# Install/sync dependencies
uv sync

# The project uses uv (not pip) for package management
# Dependencies are defined in pyproject.toml
```

### Environment Setup

Requires `.env` file in root directory:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Architecture Overview

### Request Flow

The system processes queries through multiple layers with two Claude API calls per request:

1. **Frontend** (frontend/script.js) → User input, POST to `/api/query`
2. **API Layer** (backend/app.py) → FastAPI endpoint receives request
3. **RAG System** (backend/rag_system.py) → Main orchestrator
4. **AI Generator** (backend/ai_generator.py) → **First Claude API call** with tool definitions
5. **Tool Execution** → Claude decides to use `search_course_content` tool
6. **Vector Store** (backend/vector_store.py) → ChromaDB semantic search
7. **AI Generator** → **Second Claude API call** with search results to generate final answer
8. **API Response** → Returns answer + sources to frontend

### Core Components

**RAGSystem (rag_system.py)** - Central orchestrator that:
- Coordinates document processing, vector storage, AI generation, and session management
- Implements `query()` method that handles the full request lifecycle
- Manages tool execution flow between Claude and the vector store

**AIGenerator (ai_generator.py)** - Claude API integration:
- Makes two API calls per query: first with tools, second with tool results
- Handles tool execution via `_handle_tool_execution()` method
- System prompt instructs Claude to use tools sparingly (max 1 search per query)
- Uses `claude-sonnet-4-20250514` model at temperature 0

**VectorStore (vector_store.py)** - ChromaDB wrapper with two collections:
- `course_catalog`: Stores course metadata (title, instructor, lessons) for semantic course name matching
- `course_content`: Stores text chunks (800 chars, 100 overlap) with metadata (course_title, lesson_number)
- `search()` method performs two-step search: resolve course name via semantic matching, then filter content

**Tool System (search_tools.py)**:
- `CourseSearchTool`: Exposes search capability to Claude with parameters: query (required), course_name (optional), lesson_number (optional)
- `ToolManager`: Registers tools and tracks sources from last search for UI display
- Tools follow Abstract Base Class pattern for extensibility

**DocumentProcessor (document_processor.py)** - Parses course documents:
- Expected format: Course metadata header followed by "Lesson N:" sections
- Chunks text at 800 characters with 100-character overlap for context preservation
- Creates Course and CourseChunk Pydantic models

**SessionManager (session_manager.py)**:
- Maintains conversation history (last 2 exchanges by default)
- Creates unique session IDs for conversation continuity

### Data Models (models.py)

All models are Pydantic BaseModels:
- **Course**: title (used as unique ID), course_link, instructor, lessons[]
- **Lesson**: lesson_number, title, lesson_link
- **CourseChunk**: content, course_title, lesson_number, chunk_index

### Configuration (config.py)

Key settings:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (sentence-transformers)
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results per query
- `MAX_HISTORY`: 2 conversation exchanges
- `CHROMA_PATH`: "./chroma_db" (relative to backend/)

## Key Implementation Details

### Tool Calling Pattern

The system uses Anthropic's tool calling feature with a specific pattern:

1. First API call includes tool definitions and conversation history
2. Claude analyzes the query and decides whether to use the search tool
3. If tool is used (`stop_reason="tool_use"`), execute the tool via ToolManager
4. Second API call includes original query + assistant's tool use + tool results
5. Claude synthesizes final answer from retrieved content

### Vector Search Strategy

Two-step search process:
1. If `course_name` provided: semantic search in `course_catalog` to resolve to exact title
2. Search `course_content` with filters: `course_title` (if resolved) and/or `lesson_number`
3. Returns top 5 most relevant chunks with metadata

### Source Tracking

Sources flow through multiple layers:
- `CourseSearchTool._format_results()` populates `self.last_sources`
- `ToolManager.get_last_sources()` retrieves sources after tool execution
- `RAGSystem.query()` extracts sources and resets them for next query
- API returns sources array to frontend for display

### Session Management

- Session IDs created on first query if not provided
- Conversation history formatted as string and included in Claude's system prompt
- History limited to prevent token bloat (MAX_HISTORY exchanges)

## Working with Documents

Course documents must follow this format:
```
Course Title: [Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: [Lesson Title]
Lesson Link: [Optional URL]
[Lesson content...]

Lesson 1: [Next Lesson]
[Content...]
```

Documents are loaded from `/docs` folder on application startup via `app.py` startup event.

## Frontend Architecture

Single-page application (frontend/):
- **index.html**: Layout with chat interface and sidebar (course stats, suggested questions)
- **script.js**: Handles user input, API calls, markdown rendering (marked.js), source display
- **style.css**: Styling for chat interface

Frontend uses relative API paths (`/api/*`) to work behind proxies.
- use uv to run python files
- always use uv to run the server do not use pip directly