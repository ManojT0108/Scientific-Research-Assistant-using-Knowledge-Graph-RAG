#!/usr/bin/env bash

# End-to-end pipeline runner for the Scientific Research Assistant.
# This script executes the core data and retrieval pipeline stages in order.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<USAGE
Usage:
  ./run_pipeline.sh [--skip-fetch] [--skip-download] [--skip-redis] [--query "question"] [--top-k N]

Options:
  --skip-fetch      Skip metadata fetch stage.
  --skip-download   Skip PDF download stage.
  --skip-redis      Skip Redis indexing stage.
  --query TEXT      Ask a final RAG question after pipeline completes.
  --top-k N         Top-k papers for final RAG question (default: 10).
  -h, --help        Show this help.
USAGE
}

SKIP_FETCH=0
SKIP_DOWNLOAD=0
SKIP_REDIS=0
FINAL_QUERY=""
TOP_K=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-fetch)
      SKIP_FETCH=1
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --skip-redis)
      SKIP_REDIS=1
      shift
      ;;
    --query)
      FINAL_QUERY="${2:-}"
      shift 2
      ;;
    --top-k)
      TOP_K="${2:-10}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

run_stage() {
  local title="$1"
  local cmd="$2"
  echo
  echo "============================================================"
  echo "$title"
  echo "============================================================"
  echo "Running: $cmd"
  eval "$cmd"
}

if [[ $SKIP_FETCH -eq 0 ]]; then
  run_stage "Stage 1/8 - Fetch ArXiv metadata" "python fetch_papers_metadata.py"
else
  echo "Skipping Stage 1 (fetch metadata)."
fi

if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  run_stage "Stage 2/8 - Download PDFs" "python download_pdfs.py"
else
  echo "Skipping Stage 2 (download PDFs)."
fi

run_stage "Stage 3/8 - Parse full PDFs" "python parse_full_pdfs.py"
run_stage "Stage 4/8 - Chunk full papers" "python chunk_full_papers.py"
run_stage "Stage 5/8 - Embed chunks" "python embed_chunks.py"
run_stage "Stage 6/8 - Extract entities" "python extract_entities.py"
run_stage "Stage 7/8 - Build improved knowledge graph" "python build_kg_improved.py"

if [[ $SKIP_REDIS -eq 0 ]]; then
  run_stage "Stage 8/8 - Build Redis vector index" "python redis_setup.py"
else
  echo "Skipping Stage 8 (Redis setup)."
fi

echo
if [[ -n "$FINAL_QUERY" ]]; then
  echo "Pipeline complete. Running final RAG query..."
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY is not set, cannot run enhanced_rag.py."
    exit 1
  fi
  python enhanced_rag.py "$FINAL_QUERY" --top-k "$TOP_K"
else
  echo "Pipeline complete."
  echo "Example next command:"
  echo "  python enhanced_rag.py \"How do graph neural networks improve recommendation systems?\" --top-k 10"
fi
