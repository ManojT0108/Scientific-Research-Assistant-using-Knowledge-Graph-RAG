# Scientific Research Assistant using Knowledge Graph + RAG

A research assistant pipeline that collects papers, builds semantic search + knowledge graph context, and answers researcher-style questions using retrieval augmented generation (RAG).

## Current Readiness Check

I validated this repository for clone/run readiness by:

- confirming tracked files are code-only (no generated artifacts or local virtualenv)
- checking script syntax (`python -m py_compile ...`)
- aligning pipeline file compatibility (`chunks_full.jsonl` and `chunks.jsonl` fallback)
- fixing Redis index mismatch (`arxiv_chunks_idx`)
- switching Redis host/port to env-driven defaults for container use
- removing hardcoded API keys and using `ANTHROPIC_API_KEY`

## Architecture

```mermaid
flowchart LR
    A["fetch_papers_metadata.py"] --> B["download_pdfs.py"]
    B --> C["parse_full_pdfs.py"]
    C --> D["chunk_full_papers.py"]
    D --> E["embed_chunks.py"]
    D --> F["extract_entities.py"]
    F --> G["build_kg_improved.py"]
    E --> H["redis_setup.py"]
    D --> H
    G --> I["hybrid_search.py / search_with_kg.py"]
    H --> I
    I --> J["enhanced_rag.py"]
```

### Custom Architecture Image

When you share your architecture image, place it at `docs/architecture.png` (or any path you prefer) and add it like this:

```md
![Architecture](docs/architecture.png)
```

## Repository Layout

- `ScientificResearchAssistant/` - pipeline and retrieval scripts
- `ScientificResearchAssistant/requirements.txt` - Python dependencies
- `Dockerfile` - app container image
- `docker-compose.yml` - app + Redis Stack

## Prerequisites

- Python 3.11+ (3.12 recommended)
- Redis Stack (`redis/redis-stack-server`)
- Optional (for Claude-based scripts):
  - `ANTHROPIC_API_KEY`

## Quick Start (Local)

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ScientificResearchAssistant/requirements.txt
python -m nltk.downloader punkt stopwords
cd ScientificResearchAssistant
```

Set env vars (if needed):

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export ANTHROPIC_API_KEY=your_key_here
```

## Pipeline Execution Order

Run from `ScientificResearchAssistant`:

```bash
python fetch_papers_metadata.py
python download_pdfs.py
python parse_full_pdfs.py
python chunk_full_papers.py
python embed_chunks.py
python extract_entities.py
python build_kg_improved.py
python redis_setup.py
```

Then query:

```bash
python search_redis.py "knowledge graph rag in manufacturing"
# or
python enhanced_rag.py "How do I implement contrastive learning for recommendations?" --top-k 10
```

## Comparison / Testing

Use this to benchmark semantic-only retrieval vs hybrid retrieval quality:

```bash
python evaluate_search.py
```

## Docker Usage

Build and start:

```bash
docker compose up --build -d redis
docker compose run --rm app
```

Inside container (`/workspace/ScientificResearchAssistant`):

```bash
python fetch_papers_metadata.py
python download_pdfs.py
python parse_full_pdfs.py
python chunk_full_papers.py
python embed_chunks.py
python extract_entities.py
python build_kg_improved.py
python redis_setup.py
python search_redis.py "recommendation systems"
```

## Notes

- First run will take significant time (paper download, parsing, embedding model download).
- Generated data is intentionally not committed (`ScientificResearchAssistant/data/`).
- If you run Claude-based scripts, provide `ANTHROPIC_API_KEY` via environment variable.
