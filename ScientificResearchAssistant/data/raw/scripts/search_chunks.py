import json, numpy as np, argparse, textwrap
from sentence_transformers import SentenceTransformer

EMB_PATH = "data/processed/embeddings.npy"
META_PATH = "data/processed/meta.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # same as embed step

def load_meta(path):
    meta = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def format_hit(m, score, width=110):
    title = (m.get("title") or "")[:160]
    url = m.get("url") or ""
    section = m.get("section")
    year = m.get("year")
    chunk_ix = m.get("chunk_index")
    # preview is optional; pull later if you want to store snippets
    return f"[{score:.3f}] {title}\n    {year} • {section} • chunk#{chunk_ix} • {url}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5, help="top-k results")
    p.add_argument("query", type=str, nargs="+", help="search query")
    args = p.parse_args()
    query = " ".join(args.query)

    # 1) Load embeddings and meta
    embs = np.load(EMB_PATH)                     # shape (N, D), already normalized
    meta = load_meta(META_PATH)                  # length N, aligned with embs order
    assert embs.shape[0] == len(meta), "embeddings/meta length mismatch"

    # 2) Encode query (normalize to unit length)
    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]  # (D,)

    # 3) Cosine = dot product because both sides normalized
    scores = embs @ q                            # (N,)
    topk_idx = np.argpartition(scores, -args.k)[-args.k:]
    topk_idx = topk_idx[np.argsort(scores[topk_idx])][::-1]

    print(f"\nQuery: {query}\n")
    for rank, i in enumerate(topk_idx, 1):
        print(f"{rank}. {format_hit(meta[i], float(scores[i]))}\n")

if __name__ == "__main__":
    main()