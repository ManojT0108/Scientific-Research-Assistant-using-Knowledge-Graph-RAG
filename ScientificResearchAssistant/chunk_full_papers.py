"""
Script 3: Create chunks from full papers with section metadata

Creates contextual chunks with section information
Runtime: 15-30 minutes for 1,500 papers
"""

import json
from tqdm import tqdm

def chunk_paper_sections(
    paper_id, 
    sections, 
    metadata, 
    chunk_size=500, 
    overlap=100
):
    """
    Chunk paper with section awareness
    
    Args:
        paper_id: Paper ID
        sections: Dict of section name -> text
        metadata: Paper metadata (title, year, etc.)
        chunk_size: Words per chunk
        overlap: Overlapping words between chunks
    
    Returns:
        list: Chunks with metadata
    """
    
    chunks = []
    chunk_id = 0
    
    # Section importance weights for retrieval
    section_weights = {
        'abstract': 1.0,      # Highest - overview
        'introduction': 0.7,  # Context
        'methods': 0.9,       # High - technical details
        'experiments': 0.9,   # High - datasets, metrics
        'results': 0.7,       # Good - outcomes
        'conclusion': 0.6     # Summary
    }
    
    for section_name, section_text in sections.items():
        # Skip full_text and empty sections
        if not section_text or section_name == 'full_text':
            continue
        
        # Split into words
        words = section_text.split()
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = ' '.join(words[i:i+chunk_size])
            
            # Skip tiny chunks
            if len(chunk_text) < 50:
                continue
            
            # Create chunk with rich metadata
            chunk = {
                'chunk_id': f"{paper_id}_chunk_{chunk_id}",
                'paper_id': paper_id,
                'section': section_name,
                'text': chunk_text,
                'title': metadata.get('title', ''),
                'year': metadata.get('year', 2024),
                'categories': metadata.get('categories', []),
                'url': metadata.get('url', ''),
                'section_weight': section_weights.get(section_name, 0.5)
            }
            
            chunks.append(chunk)
            chunk_id += 1
    
    return chunks

def create_all_chunks(
    parsed_file='data/processed/parsed_papers_full.jsonl',
    output_file='data/processed/chunks_full.jsonl'
):
    """
    Create chunks from all parsed papers
    
    Args:
        parsed_file: Input JSONL with parsed papers
        output_file: Output JSONL with chunks
    """
    
    print("Creating chunks from parsed papers...")
    
    total_chunks = 0
    total_papers = 0
    
    with open(output_file, 'w') as out_f:
        with open(parsed_file) as in_f:
            for line in tqdm(in_f, desc="Chunking papers"):
                parsed_paper = json.loads(line)
                
                paper_id = parsed_paper['paper_id']
                sections = parsed_paper['sections']
                
                # Metadata for chunks
                metadata = {
                    'title': parsed_paper.get('title', ''),
                    'year': parsed_paper.get('year', 2024),
                    'categories': parsed_paper.get('categories', []),
                    'url': parsed_paper.get('url', '')
                }
                
                # Create chunks for this paper
                chunks = chunk_paper_sections(paper_id, sections, metadata)
                
                # Write chunks to file
                for chunk in chunks:
                    out_f.write(json.dumps(chunk) + '\n')
                    total_chunks += 1
                
                total_papers += 1
    
    # Statistics
    avg_chunks = total_chunks / total_papers if total_papers > 0 else 0
    
    print(f"\n{'='*60}")
    print("CHUNKING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Processed: {total_papers} papers")
    print(f"📦 Created: {total_chunks:,} chunks")
    print(f"📊 Average per paper: {avg_chunks:.1f} chunks")
    print(f"📁 Output: {output_file}")
    print(f"{'='*60}")
    
    # Section distribution
    print("\nChunk distribution by section:")
    section_counts = {}
    with open(output_file) as f:
        for line in f:
            chunk = json.loads(line)
            section = chunk['section']
            section_counts[section] = section_counts.get(section, 0) + 1
    
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_chunks * 100) if total_chunks > 0 else 0
        print(f"  {section:15s}: {count:6,} ({percentage:5.1f}%)")

if __name__ == "__main__":
    print("="*60)
    print("PHASE 1.3: CREATE CONTEXTUAL CHUNKS")
    print("="*60)
    print("\nThis will create chunks with section metadata")
    print("Estimated time: 15-30 minutes")
    print("Expected output: ~22,500 chunks (~15 per paper)\n")
    
    create_all_chunks()
    
    print("\n✅ Phase 1.3 Complete!")
    print("Next: Run 4_embed_to_redis_direct.py")