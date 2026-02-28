"""
Script 2: Parse PDFs and extract sections

Extracts: abstract, introduction, methods, experiments, results, conclusion
Runtime: 1-2 hours for 1,500 PDFs
"""

import PyPDF2
import json
import os
from tqdm import tqdm
import re

def extract_sections(text):
    """
    Extract key sections from full paper text
    
    Args:
        text: Full paper text
    
    Returns:
        dict: Sections with extracted text
    """
    
    sections = {
        'abstract': '',
        'introduction': '',
        'methods': '',
        'experiments': '',
        'results': '',
        'conclusion': '',
        'full_text': text[:50000]  # Store first 50K chars
    }
    
    # Section header patterns (case-insensitive)
    patterns = {
        'abstract': r'\n\s*abstract\s*\n',
        'introduction': r'\n\s*(?:1\.?\s*)?introduction\s*\n',
        'methods': r'\n\s*(?:\d+\.?\s*)?(?:methods?|methodology|approach|model|architecture)\s*\n',
        'experiments': r'\n\s*(?:\d+\.?\s*)?(?:experiments?|experimental\s+setup|evaluation)\s*\n',
        'results': r'\n\s*(?:\d+\.?\s*)?(?:results?|findings)\s*\n',
        'conclusion': r'\n\s*(?:\d+\.?\s*)?(?:conclusion|discussion)\s*\n',
    }
    
    text_lower = text.lower()
    
    # Find section positions
    section_positions = {}
    for section_name, pattern in patterns.items():
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            section_positions[section_name] = matches[0].start()
    
    # Extract text between sections
    sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])
    
    for i, (section_name, start_pos) in enumerate(sorted_sections):
        # Find end position (start of next section or end of text)
        if i < len(sorted_sections) - 1:
            end_pos = sorted_sections[i+1][1]
        else:
            end_pos = len(text)
        
        # Extract section text
        section_text = text[start_pos:end_pos].strip()
        
        # Clean up (remove section header)
        section_text = re.sub(patterns[section_name], '', section_text, count=1, flags=re.IGNORECASE)
        
        # Limit length (10K chars per section)
        sections[section_name] = section_text[:10000]
    
    return sections

def parse_all_pdfs(
    pdf_dir='data/raw/pdfs',
    metadata_file='data/raw/arxiv_papers_metadata.jsonl',
    output_file='data/processed/parsed_papers_full.jsonl'
):
    """
    Parse all PDFs and extract sections
    
    Args:
        pdf_dir: Directory containing PDF files
        metadata_file: Path to metadata file
        output_file: Output JSONL file path
    """
    
    # Load metadata
    print("Loading metadata...")
    metadata_dict = {}
    with open(metadata_file) as f:
        for line in f:
            paper = json.loads(line)
            metadata_dict[paper['arxiv_id']] = paper
    
    # Get PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDFs to parse")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Parse PDFs
    parsed_count = 0
    failed_count = 0
    
    with open(output_file, 'w') as out_f:
        for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
            paper_id = pdf_file.replace('.pdf', '')
            pdf_path = os.path.join(pdf_dir, pdf_file)
            
            try:
                # Read PDF
                with open(pdf_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    
                    # Extract text from all pages (limit to first 50)
                    full_text = ''
                    for page in pdf_reader.pages[:50]:
                        full_text += page.extract_text()
                
                # Extract sections
                sections = extract_sections(full_text)
                
                # Get metadata
                metadata = metadata_dict.get(paper_id, {})
                
                # Create parsed paper object
                parsed_paper = {
                    'paper_id': paper_id,
                    'title': metadata.get('title', ''),
                    'year': metadata.get('year', 2024),
                    'categories': metadata.get('categories', []),
                    'url': metadata.get('url', ''),
                    'sections': sections,
                    'num_pages': len(pdf_reader.pages),
                    'text_length': len(full_text)
                }
                
                # Write to output file
                out_f.write(json.dumps(parsed_paper) + '\n')
                parsed_count += 1
                
            except Exception as e:
                # Silently skip failed PDFs (corrupted, etc.)
                failed_count += 1
                continue
    
    # Summary
    success_rate = (parsed_count / len(pdf_files) * 100) if pdf_files else 0
    
    print(f"\n{'='*60}")
    print("PARSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Parsed: {parsed_count}/{len(pdf_files)} PDFs")
    print(f"❌ Failed: {failed_count}")
    print(f"📊 Success rate: {success_rate:.1f}%")
    print(f"📁 Output: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("="*60)
    print("PHASE 1.2: PARSE PDFs WITH SECTION EXTRACTION")
    print("="*60)
    print("\nThis will parse ~1,500 PDFs and extract sections")
    print("Estimated time: 1-2 hours")
    print("Sections: abstract, intro, methods, experiments, results\n")
    
    parse_all_pdfs()
    
    print("\n✅ Phase 1.2 Complete!")
    print("Next: Run 3_chunk_full_papers.py")