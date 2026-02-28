"""
Script 1: Download PDFs for all 1,500 papers

Downloads PDFs from arXiv using existing metadata
Runtime: 30-60 minutes (with rate limiting)
"""

import json
import requests
import os
from tqdm import tqdm
import time

def download_pdfs(
    metadata_file='data/raw/arxiv_papers_metadata.jsonl',
    output_dir='data/raw/pdfs',
    delay=1.0
):
    """
    Download PDFs for all papers in metadata file
    
    Args:
        metadata_file: Path to metadata JSONL file
        output_dir: Directory to save PDFs
        delay: Seconds to wait between requests (be nice to arXiv!)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    print("Loading paper metadata...")
    papers = []
    with open(metadata_file) as f:
        for line in f:
            papers.append(json.loads(line))
    
    print(f"Found {len(papers)} papers to download")
    
    # Download PDFs
    downloaded = 0
    skipped = 0
    failed = 0

    for paper in tqdm(papers, desc="Downloading PDFs"):
        paper_id = paper['arxiv_id']
        pdf_url = paper.get('pdf_url', f"https://arxiv.org/pdf/{paper_id}.pdf")
        pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")
        
        # Skip if already exists
        if os.path.exists(pdf_path):
            skipped += 1
            continue
        
        try:
            # Download PDF
            response = requests.get(pdf_url, timeout=30)
            
            if response.status_code == 200:
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
            else:
                print(f"\nFailed: {paper_id} (HTTP {response.status_code})")
                failed += 1
            
            # Rate limiting - be respectful to arXiv
            time.sleep(delay)
            
        except Exception as e:
            print(f"\nError downloading {paper_id}: {e}")
            failed += 1
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Downloaded: {downloaded}")
    print(f"⏭️  Skipped (already exist): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total PDFs in directory: {len(os.listdir(output_dir))}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("="*60)
    print("PHASE 1.1: DOWNLOAD PDFs")
    print("="*60)
    print("\nThis will download ~1,500 PDFs from arXiv")
    print("Estimated time: 30-60 minutes")
    print("(Downloads are rate-limited to be respectful to arXiv)\n")
    
    download_pdfs()
    
    print("\n✅ Phase 1.1 Complete!")
    print("Next: Run 2_parse_pdfs_full.py")