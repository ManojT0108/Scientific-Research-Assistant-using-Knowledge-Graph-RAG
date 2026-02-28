import json, os
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

IN = "data/raw/arxiv_papers_metadata.jsonl"
OUT = "data/processed/chunks.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MAX_WORDS = 50
OVERLAP   = 30

STOP = set(stopwords.words("english"))

def chunk_sentences(text, max_words=MAX_WORDS, overlap=OVERLAP):
    """Chunk by sentences; keep ~100-word overlap to preserve continuity."""
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    chunks, cur, count = [], [], 0
    for s in sents:
        w = s.split()
        if count + len(w) > max_words:
            chunks.append(" ".join(cur).strip())
            cur = " ".join(cur).split()[-overlap:]
            count = len(cur)
        cur.extend(w); count += len(w)
    if cur: chunks.append(" ".join(cur).strip())
    return chunks

wrote = 0
skipped = 0

with open(OUT, "w", encoding="utf-8") as out:
    for line in open(IN, encoding="utf-8"):
        r = json.loads(line)
        text = (r.get("abstract") or "").strip()
        if not text:
            skipped += 1
            continue

        # Use stopwords ONLY to size chunks quickly; store the original text.
        est_len = len([w for w in text.split() if w.lower() not in STOP])
        target = 800 if est_len > 1000 else 500
        target = 50

        chunks = chunk_sentences(text, max_words=target, overlap=30)
        for i, ch in enumerate(chunks):
            rec = {
                "chunk_id": f"{r['arxiv_id']}::abs::{i}",
                "paper_id": r["arxiv_id"],
                "title": r.get("title",""),
                "year": int((r.get("published") or "0000")[:4]) if r.get("published") else None,
                "section": "abstract",
                "chunk_index": i,
                "text": ch,
                "meta": {
                    "categories": r.get("categories", []),
                    "url": r.get("pdf_url")
                }
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1

print(f"✅ Wrote {wrote} chunks to {OUT} (skipped {skipped} records with empty abstracts).")