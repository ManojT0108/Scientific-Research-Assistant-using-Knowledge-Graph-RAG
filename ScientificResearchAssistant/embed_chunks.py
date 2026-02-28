"""
Workflow Stage 5: Embed chunk text for semantic retrieval.

This script loads chunked paper text, generates sentence embeddings with
all-MiniLM-L6-v2, and writes embeddings + metadata artifacts used by Redis.
"""

import os, json, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CHUNK_CANDIDATES = [
    "data/processed/chunks_full.jsonl",
    "data/processed/chunks.jsonl",
]
EMB_PATH = "data/processed/embeddings.npy"
IDS_PATH = "data/processed/ids.jsonl"
META_PATH = "data/processed/meta.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # D=384
BATCH_SIZE = 256

os.makedirs("data/processed", exist_ok=True)

IN = next((p for p in CHUNK_CANDIDATES if os.path.exists(p)), None)
if IN is None:
    raise FileNotFoundError(
        "No chunk file found. Expected one of: "
        + ", ".join(CHUNK_CANDIDATES)
    )

# 1) Load texts + minimal display meta
texts, ids, meta = [], [], []
with open(IN, encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        ids.append(r["chunk_id"])
        texts.append(r["text"])
        meta.append({
            "chunk_id": r["chunk_id"],
            "paper_id": r.get("paper_id"),
            "title": r.get("title"),
            "year": r.get("year"),
            "section": r.get("section"),
            "chunk_index": r.get("chunk_index"),
            "url": r.get("meta", {}).get("url"),
        })

print(f"Loaded {len(texts)} chunks from {IN}")

# 2) Model
model = SentenceTransformer(MODEL_NAME)

# 3) Batch encode
embs = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
    batch = texts[i:i+BATCH_SIZE]
    vecs = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True)
    embs.append(vecs.astype("float32"))

embs = np.vstack(embs)
print(f"Embeddings shape: {embs.shape}  (rows, dim)")

# 4) Save
np.save(EMB_PATH, embs)
with open(IDS_PATH, "w", encoding="utf-8") as f:
    for cid in ids:
        f.write(json.dumps({"chunk_id": cid}) + "\n")
with open(META_PATH, "w", encoding="utf-8") as f:
    for m in meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"✅ Saved: {EMB_PATH}, {IDS_PATH}, {META_PATH}")
