import os
import re
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import spacy
from collections import defaultdict

# Load spaCy model for sentence segmentation
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[!] Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class Clause:
    """Data class to represent a document clause"""
    id: str
    text: str
    section: str
    subsection: str
    clause_type: str
    hierarchy_level: int
    parent_id: str
    line_number: int
    metadata: Dict[str, Any]

class DocumentStructureAnalyzer:
    def __init__(self):
        # Regex patterns for different document structures
        self.patterns = {
            # Section headings (various formats)
            'section_headings': [
                r'^[A-Z][A-Z\s&]+:?\s*$',  # ALL CAPS headings
                r'^\d+\.\s+[A-Z][A-Za-z\s&]+:?\s*$',  # Numbered sections
                r'^[IVX]+\.\s+[A-Z][A-Za-z\s&]+:?\s*$',  # Roman numerals
                r'^ARTICLE\s+[IVX]+\s*:?\s*[A-Z][A-Za-z\s&]*',  # ARTICLE sections
                r'^SECTION\s+\d+\s*:?\s*[A-Z][A-Za-z\s&]*',  # SECTION headings
                r'^PART\s+[A-Z]+\s*:?\s*[A-Z][A-Za-z\s&]*',  # PART headings
            ],
            
            # Numbered clauses and subclauses
            'numbered_clauses': [
                r'^\d+\.\d+\.\d+\s+',  # 1.2.3 format
                r'^\d+\.\d+\s+',       # 1.2 format
                r'^\d+\.\s+',          # 1. format
                r'^\(\d+\)\s+',        # (1) format
                r'^[a-z]\)\s+',        # a) format
                r'^\([a-z]\)\s+',      # (a) format
            ],
            
            # Bullet points and lists
            'bullet_points': [
                r'^[•·▪▫‣⁃]\s+',       # Various bullet symbols
                r'^-\s+',              # Dash bullets
                r'^\*\s+',             # Asterisk bullets
            ],
            
            # Special insurance document patterns
            'insurance_specific': [
                r'^Coverage\s*:?\s*',
                r'^Exclusion\s*:?\s*',
                r'^Condition\s*:?\s*',
                r'^Deductible\s*:?\s*',
                r'^Limit\s*:?\s*',
                r'^Premium\s*:?\s*',
                r'^Policy\s+Period\s*:?\s*',
            ],
            
            # Definitions and terms
            'definitions': [
                r'^"[^"]+"\s+means\s+',
                r'^[A-Z][A-Za-z\s]+\s+means\s+',
                r'^DEFINITION\s*:?\s*',
            ],
        }
        
        self.clause_counter = 0
        
    def extract_text_from_file(self, file_path: str) -> str:
        """Read extracted text from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[!] Error reading file {file_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page numbers and headers/footers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip likely page numbers, headers, footers
            if (re.match(r'^\d+$', line) or 
                re.match(r'^Page\s+\d+', line, re.IGNORECASE) or
                len(line) < 3):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def identify_section_headings(self, lines: List[str]) -> Dict[int, Dict[str, Any]]:
        """Identify section headings in the document"""
        headings = {}
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            for pattern in self.patterns['section_headings']:
                if re.match(pattern, line):
                    headings[i] = {
                        'text': line,
                        'type': 'section_heading',
                        'level': self._determine_heading_level(line),
                        'pattern_matched': pattern
                    }
                    break
        
        return headings
    
    def _determine_heading_level(self, heading: str) -> int:
        """Determine the hierarchical level of a heading"""
        heading = heading.strip()
        
        # ARTICLE level (highest)
        if re.match(r'^ARTICLE', heading, re.IGNORECASE):
            return 1
        # PART level
        elif re.match(r'^PART', heading, re.IGNORECASE):
            return 2
        # SECTION level
        elif re.match(r'^SECTION', heading, re.IGNORECASE):
            return 3
        # Numbered sections
        elif re.match(r'^\d+\.', heading):
            return 4
        # All caps headings
        elif heading.isupper():
            return 3
        else:
            return 5
    
    def identify_clauses(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Identify individual clauses in the document"""
        clauses = []
        current_section = "PREAMBLE"
        current_subsection = ""
        section_headings = self.identify_section_headings(lines)
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section heading
            if i in section_headings:
                current_section = self._extract_section_name(line)
                current_subsection = ""
                continue
            
            clause_info = self._analyze_clause(line, i)
            if clause_info:
                self.clause_counter += 1
                clause_data = {
                    'id': f"clause_{self.clause_counter:04d}",
                    'text': line,
                    'section': current_section,
                    'subsection': current_subsection,
                    'clause_type': clause_info['type'],
                    'hierarchy_level': clause_info['level'],
                    'line_number': i,
                    'metadata': clause_info['metadata']
                }
                clauses.append(clause_data)
                
                # Update subsection if this is a subsection header
                if clause_info['type'] in ['numbered_clause', 'subsection_header']:
                    current_subsection = self._extract_subsection_name(line)
        
        return clauses
    
    def _extract_section_name(self, heading: str) -> str:
        """Extract clean section name from heading"""
        # Remove common prefixes and clean up
        heading = re.sub(r'^(ARTICLE|SECTION|PART)\s+[IVX\d]+\s*:?\s*', '', heading, flags=re.IGNORECASE)
        heading = re.sub(r'^\d+\.\s*', '', heading)
        heading = heading.strip(':').strip()
        return heading if heading else "UNNAMED_SECTION"
    
    def _extract_subsection_name(self, text: str) -> str:
        """Extract subsection name from clause text"""
        # Take first few words as subsection name
        words = text.split()[:5]
        return ' '.join(words) if words else "UNNAMED_SUBSECTION"
    
    def _analyze_clause(self, line: str, line_number: int) -> Dict[str, Any]:
        """Analyze a line to determine if it's a clause and its properties"""
        metadata = {}
        
        # Check for numbered clauses
        for pattern in self.patterns['numbered_clauses']:
            match = re.match(pattern, line)
            if match:
                return {
                    'type': 'numbered_clause',
                    'level': self._get_numbering_level(match.group()),
                    'metadata': {'numbering': match.group().strip(), **metadata}
                }
        
        # Check for bullet points
        for pattern in self.patterns['bullet_points']:
            if re.match(pattern, line):
                return {
                    'type': 'bullet_point',
                    'level': 6,
                    'metadata': metadata
                }
        
        # Check for insurance-specific patterns
        for pattern in self.patterns['insurance_specific']:
            if re.match(pattern, line, re.IGNORECASE):
                return {
                    'type': 'insurance_clause',
                    'level': 4,
                    'metadata': {'insurance_type': self._extract_insurance_type(line), **metadata}
                }
        
        # Check for definitions
        for pattern in self.patterns['definitions']:
            if re.match(pattern, line, re.IGNORECASE):
                return {
                    'type': 'definition',
                    'level': 5,
                    'metadata': {'defined_term': self._extract_defined_term(line), **metadata}
                }
        
        # Default: treat as paragraph if it's substantial
        if len(line.split()) > 5:
            return {
                'type': 'paragraph',
                'level': 7,
                'metadata': metadata
            }
        
        return None
    
    def _get_numbering_level(self, numbering: str) -> int:
        """Determine hierarchical level based on numbering format"""
        numbering = numbering.strip()
        
        if re.match(r'\d+\.\d+\.\d+', numbering):
            return 6  # 1.2.3 level
        elif re.match(r'\d+\.\d+', numbering):
            return 5  # 1.2 level
        elif re.match(r'\d+\.', numbering):
            return 4  # 1. level
        elif re.match(r'\(\d+\)', numbering):
            return 5  # (1) level
        elif re.match(r'[a-z]\)', numbering):
            return 6  # a) level
        elif re.match(r'\([a-z]\)', numbering):
            return 6  # (a) level
        else:
            return 7
    
    def _extract_insurance_type(self, line: str) -> str:
        """Extract the type of insurance clause"""
        line_lower = line.lower()
        if 'coverage' in line_lower:
            return 'coverage'
        elif 'exclusion' in line_lower:
            return 'exclusion'
        elif 'condition' in line_lower:
            return 'condition'
        elif 'deductible' in line_lower:
            return 'deductible'
        elif 'limit' in line_lower:
            return 'limit'
        elif 'premium' in line_lower:
            return 'premium'
        else:
            return 'general'
    
    def _extract_defined_term(self, line: str) -> str:
        """Extract the term being defined"""
        # Look for quoted terms
        quote_match = re.search(r'"([^"]+)"', line)
        if quote_match:
            return quote_match.group(1)
        
        # Look for terms before "means"
        means_match = re.search(r'^([^"]+?)\s+means\s+', line, re.IGNORECASE)
        if means_match:
            return means_match.group(1).strip()
        
        return "UNKNOWN_TERM"
    
    def establish_hierarchy(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Establish parent-child relationships between clauses"""
        for i, clause in enumerate(clauses):
            parent_id = ""
            current_level = clause['hierarchy_level']
            
            # Look backwards for a parent (higher level)
            for j in range(i-1, -1, -1):
                if clauses[j]['hierarchy_level'] < current_level:
                    parent_id = clauses[j]['id']
                    break
            
            clause['parent_id'] = parent_id
        
        return clauses
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using spaCy"""
        if not nlp:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if s.strip()]
        
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def process_document(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Main function to process a single document"""
        print(f"[+] Analyzing structure of: {os.path.basename(input_path)}")
        
        # Extract and clean text
        raw_text = self.extract_text_from_file(input_path)
        if not raw_text:
            return {}
        
        cleaned_text = self.clean_text(raw_text)
        lines = cleaned_text.split('\n')
        
        # Identify clauses and structure
        clauses_data = self.identify_clauses(lines)
        clauses_with_hierarchy = self.establish_hierarchy(clauses_data)
        
        # Create structured output
        result = {
            'document_info': {
                'source_file': os.path.basename(input_path),
                'total_clauses': len(clauses_with_hierarchy),
                'processing_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else str(datetime.now())
            },
            'structure_summary': self._generate_structure_summary(clauses_with_hierarchy),
            'clauses': clauses_with_hierarchy
        }
        
        # Save structured output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
    
    def _generate_structure_summary(self, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics about document structure"""
        summary = {
            'clause_types': defaultdict(int),
            'sections': set(),
            'hierarchy_levels': defaultdict(int),
            'insurance_clauses': defaultdict(int)
        }
        
        for clause in clauses:
            summary['clause_types'][clause['clause_type']] += 1
            summary['sections'].add(clause['section'])
            summary['hierarchy_levels'][clause['hierarchy_level']] += 1
            
            if clause['clause_type'] == 'insurance_clause':
                insurance_type = clause['metadata'].get('insurance_type', 'unknown')
                summary['insurance_clauses'][insurance_type] += 1
        
        # Convert sets to lists for JSON serialization
        summary['sections'] = list(summary['sections'])
        summary['clause_types'] = dict(summary['clause_types'])
        summary['hierarchy_levels'] = dict(summary['hierarchy_levels'])
        summary['insurance_clauses'] = dict(summary['insurance_clauses'])
        
        return summary

def process_all_extracted_files(input_dir: str = "output/extracted_text", 
                               output_dir: str = "output/structured_analysis"):
    """Process all extracted text files and generate structured analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = DocumentStructureAnalyzer()
    results_summary = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{filename}_structured.json")
            
            result = analyzer.process_document(input_path, output_path)
            if result:
                results_summary.append({
                    'file': filename,
                    'total_clauses': result['document_info']['total_clauses'],
                    'structure_summary': result['structure_summary']
                })
                print(f"[✓] Processed: {filename} -> {result['document_info']['total_clauses']} clauses")
    
    # Save overall summary
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_documents': len(results_summary),
            'processing_details': results_summary
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[✔] Structure analysis complete! Processed {len(results_summary)} documents.")
    print(f"[✔] Results saved in: {output_dir}")
    return results_summary

if __name__ == "__main__":
    # Import required libraries
    try:
        import pandas as pd
    except ImportError:
        pass
    
    from datetime import datetime
    
    # Process all extracted files
    results = process_all_extracted_files()
    
    # Print summary
    print("\n" + "="*50)
    print("DOCUMENT STRUCTURE ANALYSIS SUMMARY")
    print("="*50)
    for result in results:
        print(f"File: {result['file']}")
        print(f"  Total Clauses: {result['total_clauses']}")
        print(f"  Sections: {len(result['structure_summary']['sections'])}")
        print(f"  Clause Types: {list(result['structure_summary']['clause_types'].keys())}")
        print()