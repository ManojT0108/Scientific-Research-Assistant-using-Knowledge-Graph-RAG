# Scientific Research Assistant using Knowledge Graph + RAG

A research assistant pipeline that collects papers, builds semantic search + knowledge graph context, and answers researcher-style questions using retrieval augmented generation (RAG).

## What This Project Does

The system follows this workflow:

1. Fetch CS paper metadata from ArXiv
2. Download full PDFs and parse section-aware text
3. Chunk and embed papers for semantic retrieval
4. Build a knowledge graph from extracted entities
5. Compare semantic-only retrieval vs hybrid (semantic + graph) retrieval
6. Generate natural-language answers using retrieved evidence and an LLM

## Why Knowledge Graph Helps

Vector search is strong at finding text that is semantically similar to a query, but it can miss papers that are highly relevant and use different wording.  
The knowledge graph adds a second retrieval path based on research entities and relationships.

### Problem With Semantic-Only Retrieval

In research literature, two papers can be strongly related even if they do not share many similar phrases.  
For example, one paper might discuss "contrastive representation learning for recommendation" and another might use different terminology but still solve the same task with related methods and datasets.

### What The Graph Represents

The graph links each paper to extracted entities such as:

- tasks (for example, recommendation, question answering)
- datasets (for example, MovieLens, SQuAD)
- metrics (for example, NDCG, F1)
- methods (for example, Transformer, GNN)

Papers become connected through shared entities.  
This gives the system explicit relational context that embeddings alone do not expose.

### Hybrid Retrieval Strategy

The project uses a hybrid approach:

1. Run vector retrieval in Redis to get top semantically similar papers.
2. Expand from those seed papers through graph connections.
3. Re-rank results by combining vector similarity and graph relevance.

The final result set is usually broader and more useful for literature exploration than semantic-only retrieval.

### Why Filtering Matters

Not all entities are informative. Very common entities (for example, generic terms that appear everywhere) can create noisy graph links.  
The pipeline addresses this by weighting/filtering entity influence so specific entities contribute more than generic ones.

### Effect On LLM Answers

The LLM does not just receive "closest chunks"; it receives evidence that includes semantically relevant papers plus graph-connected papers.  
This improves the chance of:

- better method comparisons
- stronger dataset recommendations
- richer multi-paper synthesis
- more complete researcher-facing answers

## Architecture

![Knowledge Graph Structure](docs/architecture.png)

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

Or run the full workflow with one command:

```bash
./run_pipeline.sh
```

Optional usage:

```bash
./run_pipeline.sh --query "How do graph neural networks improve recommendation systems?" --top-k 10
./run_pipeline.sh --skip-fetch --skip-download
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
