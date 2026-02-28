"""
Workflow Stage 1: Fetch ArXiv metadata for CS paper cohorts.

Pulls metadata for configured categories/year and writes a JSONL catalog
that downstream scripts use for PDF download and full-text parsing.
"""

import arxiv, json, os, time, calendar

CATEGORIES = ["cs.IR", "cs.DB", "cs.LG"]
YEAR = 2024
MAX_PER_CAT = 500
PAGE_SIZE = 100
DELAY = 2
OUT = "data/raw/arxiv_papers_metadata.jsonl"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

def warm_seen(path):
    seen=set()
    if os.path.exists(path):
        with open(path,"r",encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec=json.loads(line)
                    if rec.get("arxiv_id"): seen.add(rec["arxiv_id"])
                except: pass
    return seen

def yyyymmddhhmm(y,m,d,hh=0,mm=0):
    return f"{y:04d}{m:02d}{d:02d}{hh:02d}{mm:02d}"

def write(outfh, r):
    outfh.write(json.dumps({
        "arxiv_id": r.get_short_id(),
        "title": (r.title or "").strip(),
        "abstract": (r.summary or "").strip(),
        "categories": list(r.categories) if r.categories else [],
        "published": r.published.date().isoformat() if r.published else None,
        "updated": r.updated.date().isoformat() if r.updated else None,
        "pdf_url": r.pdf_url
    }, ensure_ascii=False) + "\n")

def harvest():
    seen = warm_seen(OUT)
    client = arxiv.Client(page_size=PAGE_SIZE, delay_seconds=DELAY, num_retries=6)
    total_new = 0

    with open(OUT, "a", encoding="utf-8") as out:
        for cat in CATEGORIES:
            cat_count = 0
            print(f"\n== {cat} {YEAR} (target {MAX_PER_CAT}) ==")
            for month in range(1, 13):
                if cat_count >= MAX_PER_CAT: break
                last_day = calendar.monthrange(YEAR, month)[1]
                start = yyyymmddhhmm(YEAR, month, 1, 0, 0)                 
                end   = yyyymmddhhmm(YEAR, month, last_day, 23, 59)      
                q = f"cat:{cat} AND submittedDate:[{start} TO {end}]"
                print(f"  {YEAR}-{month:02d} :: fetching…")
                try:
                    search = arxiv.Search(query=q, max_results=PAGE_SIZE,
                                          sort_by=arxiv.SortCriterion.SubmittedDate)
                    for r in client.results(search):
                        if not r.published or r.published.date().year != YEAR:
                            continue
                        sid = r.get_short_id()
                        if sid in seen:
                            continue
                        write(out, r)
                        seen.add(sid); total_new += 1; cat_count += 1
                        if cat_count % 100 == 0:
                            print(f"    {cat}: {cat_count} so far")
                        if cat_count >= MAX_PER_CAT: break
                        time.sleep(0.03)
                except Exception as e:
                    print(f"    ⚠️ {YEAR}-{month:02d} window error: {e}; continuing…")
            print(f"→ {cat} collected: {cat_count}")

    print(f"\n✅ Done. New this run: {total_new}")
    print(f"→ Output: {OUT}")

if __name__ == "__main__":
    harvest()
