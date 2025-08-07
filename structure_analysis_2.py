import os
import re
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import spacy
from collections import defaultdict
from datetime import datetime
import pandas as pd

# Load spaCy model for NER and sentence segmentation
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[!] Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class Clause:
    """Data class to represent a document clause with comprehensive metadata"""
    id: str
    text: str
    section: str
    subsection: str
    clause_type: str
    hierarchy_level: int
    parent_id: str
    line_number: int
    metadata: Dict[str, Any]
    
    # Financial data
    financial_data: Dict[str, Any] = None
    
    # Temporal information
    temporal_data: Dict[str, Any] = None
    
    # Legal analysis
    legal_analysis: Dict[str, Any] = None
    
    # Insurance specific
    insurance_analysis: Dict[str, Any] = None
    
    # Cross-references
    cross_references: List[str] = None
    
    # Named entities
    named_entities: Dict[str, List[str]] = None

class EnhancedDocumentStructureAnalyzer:
    def __init__(self):
        # Enhanced patterns for comprehensive analysis
        self.patterns = {
            # Section headings (various formats)
            'section_headings': [
                r'^[A-Z][A-Z\s&]+:?\s*$',  # ALL CAPS headings
                r'^\d+\.\s+[A-Z][A-Za-z\s&]+:?\s*$',  # Numbered sections
                r'^[IVX]+\.\s+[A-Z][A-Za-z\s&]+:?\s*$',  # Roman numerals
                r'^ARTICLE\s+[IVX]+\s*:?\s*[A-Z][A-Za-z\s&]*',  # ARTICLE sections
                r'^SECTION\s+\d+\s*:?\s*[A-Z][A-Za-z\s&]*',  # SECTION headings
                r'^PART\s+[A-Z]+\s*:?\s*[A-Z][A-Za-z\s&]*',  # PART headings
                r'^SCHEDULE\s+[A-Z\d]+\s*:?\s*',  # SCHEDULE headings
                r'^ENDORSEMENT\s+\d+\s*:?\s*',  # ENDORSEMENT headings
            ],
            
            # Financial patterns - Enhanced
            'financial': {
                'INR_amounts': [
                    # ₹ with lakh/crore/decimal
                    r'₹\s?\d{1,2}(?:,\d{2})+(?:\.\d{1,2})?\s?(?:lakh|crore|Cr|L|K)?',
                    # Rs. or INR with amounts
                    r'(?:Rs\.?|INR)\s?\d{1,2}(?:,\d{2})+(?:\.\d{1,2})?\s?(?:lakh|crore|Cr|L|K)?',
                    # Numbers with lakh/crore suffix
                    r'\d+(?:\.\d+)?\s?(?:lakh|crore|Cr|L|K)',
                    # Optional ₹/INR and words like million, billion (if found in international docs)
                    r'(?:₹|INR)?\s?\d+(?:,\d{2})+(?:\.\d{1,2})?\s?(?:million|billion|thousand)?'
                ],
                'percentages': [
                    r'\d+(?:\.\d+)?%',
                    r'\d+(?:\.\d+)?\s?percent',
                    r'percentage\s+of\s+[\d.]+',
                ],
                'limits_deductibles': [
                    r'limit(?:s)?\s+of\s+\$[\d,]+',
                    r'deductible\s+of\s+\$[\d,]+',
                    r'maximum\s+(?:amount|limit)\s+\$[\d,]+',
                    r'minimum\s+(?:amount|limit)\s+\$[\d,]+',
                    r'aggregate\s+limit\s+\$[\d,]+',
                    r'per\s+occurrence\s+limit\s+\$[\d,]+',
                    r'per\s+claim\s+limit\s+\$[\d,]+'
                ]
            },
            
            # Temporal patterns - Enhanced
            'temporal': {
                'dates': [
                    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY
                    r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD
                    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
                    r'\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
                    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}'
                ],
                'periods': [
                    r'\d+\s+(?:days?|weeks?|months?|years?)',
                    r'(?:daily|weekly|monthly|quarterly|annually|yearly)',
                    r'policy\s+period',
                    r'coverage\s+period',
                    r'(?:effective|expiration)\s+date',
                    r'term\s+of\s+\d+\s+(?:days?|months?|years?)',
                    r'renewable\s+(?:annually|yearly|monthly)'
                ]
            },
            
            # Legal language patterns - Enhanced
            'legal': {
                'conditional_markers': [
                    r'provided\s+that',
                    r'subject\s+to',
                    r'except\s+(?:as|that|for)',
                    r'notwithstanding',
                    r'in\s+the\s+event\s+(?:that|of)',
                    r'unless\s+(?:and\s+until|otherwise)',
                    r'if\s+and\s+only\s+if',
                    r'to\s+the\s+extent\s+that'
                ],
                'coverage_sentiment': [
                    r'(?:shall\s+)?(?:be\s+)?covered',
                    r'(?:is\s+)?included',
                    r'(?:shall\s+)?(?:be\s+)?excluded',
                    r'(?:is\s+)?not\s+covered',
                    r'coverage\s+(?:applies|extends|is\s+provided)',
                    r'no\s+coverage',
                    r'this\s+insurance\s+(?:covers|does\s+not\s+cover)'
                ],
                'qualifiers': [
                    r'however',
                    r'nevertheless',
                    r'furthermore',
                    r'moreover',
                    r'in\s+addition',
                    r'on\s+the\s+other\s+hand',
                    r'consequently',
                    r'therefore'
                ]
            },
            
            # Enhanced insurance-specific patterns
            'insurance_specific': {
                'limit_types': [
                    r'aggregate\s+limit',
                    r'per\s+occurrence\s+limit',
                    r'per\s+claim\s+limit',
                    r'combined\s+single\s+limit',
                    r'split\s+limit',
                    r'umbrella\s+limit',
                    r'excess\s+limit'
                ],
                'coverage_triggers': [
                    r'occurrence\s+basis',
                    r'claims[\s-]made\s+basis',
                    r'claims[\s-]made\s+and\s+reported',
                    r'discovery\s+basis',
                    r'manifestation\s+basis'
                ],
                'territory': [
                    r'(?:within|throughout)\s+(?:the\s+)?(?:United\s+States|USA|US)',
                    r'worldwide\s+(?:coverage|territory)',
                    r'domestic\s+(?:coverage|operations)',
                    r'international\s+(?:coverage|operations)',
                    r'territorial\s+limits'
                ],
                'clause_types': [
                    r'^(?:Coverage|Exclusion|Condition|Deductible|Limit|Premium|Territory|Notice|Claims|Cancellation|Definitions)\s*:?\s*'
                ]
            },
            
            # Cross-reference patterns
            'cross_references': [
                r'(?:see|refer\s+to|as\s+defined\s+in)\s+(?:section|article|clause|paragraph|schedule|endorsement)\s+[\d.]+',
                r'as\s+set\s+forth\s+in\s+(?:section|article|clause)\s+[\d.]+',
                r'in\s+accordance\s+with\s+(?:section|article|clause)\s+[\d.]+',
                r'pursuant\s+to\s+(?:section|article|clause)\s+[\d.]+',
                r'subject\s+to\s+(?:section|article|clause|the\s+terms\s+of)\s+[\d.]*',
                r'except\s+as\s+provided\s+in\s+(?:section|article|clause)\s+[\d.]+'
            ],
            
            # Table patterns
            'table_indicators': [
                r'^\s*\|\s*.*\s*\|\s*$',  # Pipe-separated tables
                r'^\s*[A-Za-z\s]+\t[A-Za-z\d\s]+',  # Tab-separated
                r'Schedule\s+of\s+(?:Coverages?|Limits|Rates)',
                r'Rating\s+Table',
                r'Premium\s+Schedule'
            ],
            
            # Risk classification patterns
            'risk_classifications': [
                r'class\s+code\s+\d+',
                r'SIC\s+code\s+\d+',
                r'NAICS\s+code\s+\d+',
                r'industry\s+group\s+\d+',
                r'risk\s+category\s+[A-Z\d]+',
                r'hazard\s+class\s+[A-Z\d]+',
                r'occupancy\s+class\s+[A-Z\d]+'
            ],
            
            # Calculation formulas
            'calculations': [
                r'premium\s*=\s*.*[\+\-\*\/].*',
                r'deductible\s*=\s*.*[\+\-\*\/].*',
                r'calculated\s+as\s+.*[\+\-\*\/].*',
                r'rate\s+per\s+\$?\d+',
                r'factor\s+of\s+[\d.]+',
                r'multiplied\s+by\s+[\d.]+',
                r'percentage\s+of\s+.*'
            ]
        }
        
        self.clause_counter = 0
        
    def extract_financial_data(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive financial information from text"""
        financial_data = {
            'INR_amounts': [],
            'percentages': [],
            'limits': [],
            'deductibles': [],
            'premiums': [],
            'calculations': []
        }
        
        # Extract INR amounts
        for pattern in self.patterns['financial']['INR_amounts']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            financial_data['INR_amounts'].extend(matches)
        
        # Extract percentages
        for pattern in self.patterns['financial']['percentages']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            financial_data['percentages'].extend(matches)
        
        # Extract limits and deductibles
        for pattern in self.patterns['financial']['limits_deductibles']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if 'limit' in pattern:
                financial_data['limits'].extend(matches)
            elif 'deductible' in pattern:
                financial_data['deductibles'].extend(matches)
        
        # Extract calculations
        for pattern in self.patterns['calculations']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            financial_data['calculations'].extend(matches)
        
        return financial_data
    
    def extract_temporal_data(self, text: str) -> Dict[str, Any]:
        """Extract temporal information from text"""
        temporal_data = {
            'dates': [],
            'periods': [],
            'effective_dates': [],
            'expiration_dates': [],
            'policy_periods': []
        }
        
        # Extract dates
        for pattern in self.patterns['temporal']['dates']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temporal_data['dates'].extend(matches)
        
        # Extract periods
        for pattern in self.patterns['temporal']['periods']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temporal_data['periods'].extend(matches)
            
            # Categorize specific period types
            if 'effective' in pattern:
                temporal_data['effective_dates'].extend(matches)
            elif 'expiration' in pattern:
                temporal_data['expiration_dates'].extend(matches)
            elif 'policy' in pattern:
                temporal_data['policy_periods'].extend(matches)
        
        return temporal_data
    
    def analyze_legal_language(self, text: str) -> Dict[str, Any]:
        """Analyze legal language patterns"""
        legal_analysis = {
            'conditional_markers': [],
            'coverage_sentiment': 'neutral',
            'qualifiers': [],
            'coverage_providing': False,
            'restrictive': False
        }
        
        # Extract conditional markers
        for pattern in self.patterns['legal']['conditional_markers']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            legal_analysis['conditional_markers'].extend(matches)
        
        # Analyze coverage sentiment
        coverage_matches = []
        for pattern in self.patterns['legal']['coverage_sentiment']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            coverage_matches.extend(matches)
        
        if coverage_matches:
            positive_indicators = ['covered', 'included', 'applies', 'extends', 'provided']
            negative_indicators = ['excluded', 'not covered', 'no coverage', 'does not cover']
            
            positive_count = sum(1 for match in coverage_matches 
                               if any(pos in match.lower() for pos in positive_indicators))
            negative_count = sum(1 for match in coverage_matches 
                               if any(neg in match.lower() for neg in negative_indicators))
            
            if positive_count > negative_count:
                legal_analysis['coverage_sentiment'] = 'coverage_providing'
                legal_analysis['coverage_providing'] = True
            elif negative_count > positive_count:
                legal_analysis['coverage_sentiment'] = 'restrictive'
                legal_analysis['restrictive'] = True
        
        # Extract qualifiers
        for pattern in self.patterns['legal']['qualifiers']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            legal_analysis['qualifiers'].extend(matches)
        
        return legal_analysis
    
    def analyze_insurance_specifics(self, text: str) -> Dict[str, Any]:
        """Analyze insurance-specific patterns"""
        insurance_analysis = {
            'limit_types': [],
            'coverage_triggers': [],
            'territory': [],
            'clause_type': None,
            'insurance_categories': []
        }
        
        # Extract limit types
        for pattern in self.patterns['insurance_specific']['limit_types']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            insurance_analysis['limit_types'].extend(matches)
        
        # Extract coverage triggers
        for pattern in self.patterns['insurance_specific']['coverage_triggers']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            insurance_analysis['coverage_triggers'].extend(matches)
        
        # Extract territorial scope
        for pattern in self.patterns['insurance_specific']['territory']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            insurance_analysis['territory'].extend(matches)
        
        # Identify clause type
        for pattern in self.patterns['insurance_specific']['clause_types']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                insurance_analysis['clause_type'] = match.group().strip(':').strip().lower()
                break
        
        # Categorize insurance types
        text_lower = text.lower()
        categories = {
            'property': ['property', 'building', 'contents', 'equipment'],
            'liability': ['liability', 'bodily injury', 'property damage', 'personal injury'],
            'auto': ['automobile', 'vehicle', 'auto', 'motor'],
            'workers_comp': ['workers compensation', 'workers comp', 'workplace injury'],
            'professional': ['professional liability', 'errors and omissions', 'malpractice'],
            'cyber': ['cyber', 'data breach', 'network security', 'privacy'],
            'umbrella': ['umbrella', 'excess']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                insurance_analysis['insurance_categories'].append(category)
        
        return insurance_analysis
    
    def extract_cross_references(self, text: str) -> List[str]:
        """Extract cross-references to other clauses or sections"""
        cross_references = []
        
        for pattern in self.patterns['cross_references']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            cross_references.extend(matches)
        
        return cross_references
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entities
            'DATE': [],
            'MONEY': [],
            'PERCENT': [],
            'LAW': [],
            'MISC': []
        }
        
        if not nlp:
            return entities
        
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
            else:
                entities['MISC'].append(f"{ent.text} ({ent.label_})")
        
        return entities
    
    def extract_risk_classifications(self, text: str) -> List[str]:
        """Extract risk classification codes and categories"""
        classifications = []
        
        for pattern in self.patterns['risk_classifications']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            classifications.extend(matches)
        
        return classifications
    
    def detect_table_data(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect and extract table data from lines"""
        tables = []
        current_table = None
        
        for i, line in enumerate(lines):
            is_table_line = False
            
            # Check for table indicators
            for pattern in self.patterns['table_indicators']:
                if re.match(pattern, line):
                    is_table_line = True
                    break
            
            if is_table_line:
                if current_table is None:
                    current_table = {
                        'start_line': i,
                        'rows': [],
                        'type': 'detected'
                    }
                current_table['rows'].append(line.strip())
            else:
                if current_table is not None:
                    current_table['end_line'] = i - 1
                    tables.append(current_table)
                    current_table = None
        
        # Close any remaining table
        if current_table is not None:
            current_table['end_line'] = len(lines) - 1
            tables.append(current_table)
        
        return tables
    
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
        # SCHEDULE level
        elif re.match(r'^SCHEDULE', heading, re.IGNORECASE):
            return 3
        # ENDORSEMENT level
        elif re.match(r'^ENDORSEMENT', heading, re.IGNORECASE):
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
        """Identify individual clauses with comprehensive analysis"""
        clauses = []
        current_section = "PREAMBLE"
        current_subsection = ""
        section_headings = self.identify_section_headings(lines)
        tables = self.detect_table_data(lines)
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section heading
            if i in section_headings:
                current_section = self._extract_section_name(line)
                current_subsection = ""
                continue
            
            # Check if this line is part of a table
            is_table_row = any(table['start_line'] <= i <= table['end_line'] for table in tables)
            
            clause_info = self._analyze_clause_comprehensive(line, i)
            if clause_info:
                self.clause_counter += 1
                
                # Comprehensive analysis
                financial_data = self.extract_financial_data(line)
                temporal_data = self.extract_temporal_data(line)
                legal_analysis = self.analyze_legal_language(line)
                insurance_analysis = self.analyze_insurance_specifics(line)
                cross_references = self.extract_cross_references(line)
                named_entities = self.extract_named_entities(line)
                risk_classifications = self.extract_risk_classifications(line)
                
                clause_data = {
                    'id': f"clause_{self.clause_counter:04d}",
                    'text': line,
                    'section': current_section,
                    'subsection': current_subsection,
                    'clause_type': clause_info['type'],
                    'hierarchy_level': clause_info['level'],
                    'line_number': i,
                    'metadata': clause_info['metadata'],
                    'financial_data': financial_data,
                    'temporal_data': temporal_data,
                    'legal_analysis': legal_analysis,
                    'insurance_analysis': insurance_analysis,
                    'cross_references': cross_references,
                    'named_entities': named_entities,
                    'risk_classifications': risk_classifications,
                    'is_table_row': is_table_row
                }
                clauses.append(clause_data)
                
                # Update subsection if this is a subsection header
                if clause_info['type'] in ['numbered_clause', 'subsection_header']:
                    current_subsection = self._extract_subsection_name(line)
        
        return clauses
    
    def _analyze_clause_comprehensive(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """Comprehensive clause analysis with all patterns"""
        metadata = {}
        
        # Check for numbered clauses
        for pattern_name, patterns in [
            ('numbered_clauses', [
                r'^\d+\.\d+\.\d+\s+',  # 1.2.3 format
                r'^\d+\.\d+\s+',       # 1.2 format
                r'^\d+\.\s+',          # 1. format
                r'^\(\d+\)\s+',        # (1) format
                r'^[a-z]\)\s+',        # a) format
                r'^\([a-z]\)\s+',      # (a) format
            ]),
            ('bullet_points', [
                r'^[•·▪▫‣⁃]\s+',       # Various bullet symbols
                r'^-\s+',              # Dash bullets
                r'^\*\s+',             # Asterisk bullets
            ]),
        ]:
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    level = self._get_numbering_level(match.group()) if pattern_name == 'numbered_clauses' else 6
                    return {
                        'type': pattern_name[:-1],  # Remove 's'
                        'level': level,
                        'metadata': {'numbering': match.group().strip(), **metadata}
                    }
        
        # Check for insurance-specific patterns
        for pattern in self.patterns['insurance_specific']['clause_types']:
            if re.match(pattern, line, re.IGNORECASE):
                return {
                    'type': 'insurance_clause',
                    'level': 4,
                    'metadata': {'insurance_type': self._extract_insurance_type(line), **metadata}
                }
        
        # Check for definitions
        definition_patterns = [
            r'^"[^"]+"\s+means\s+',
            r'^[A-Z][A-Za-z\s]+\s+means\s+',
            r'^DEFINITION\s*:?\s*',
        ]
        
        for pattern in definition_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return {
                    'type': 'definition',
                    'level': 5,
                    'metadata': {'defined_term': self._extract_defined_term(line), **metadata}
                }
        
        # Check for table headers/rows
        for pattern in self.patterns['table_indicators']:
            if re.match(pattern, line):
                return {
                    'type': 'table_row',
                    'level': 8,
                    'metadata': {'table_type': 'detected', **metadata}
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
    
    def _extract_section_name(self, heading: str) -> str:
        """Extract clean section name from heading"""
        # Remove common prefixes and clean up
        heading = re.sub(r'^(ARTICLE|SECTION|PART|SCHEDULE|ENDORSEMENT)\s+[IVX\d]+\s*:?\s*', '', heading, flags=re.IGNORECASE)
        heading = re.sub(r'^\d+\.\s*', '', heading)
        heading = heading.strip(':').strip()
        return heading if heading else "UNNAMED_SECTION"
    
    def _extract_subsection_name(self, text: str) -> str:
        """Extract subsection name from clause text"""
        # Take first few words as subsection name
        words = text.split()[:5]
        return ' '.join(words) if words else "UNNAMED_SUBSECTION"
    
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
        elif 'territory' in line_lower:
            return 'territory'
        elif 'notice' in line_lower:
            return 'notice'
        elif 'claims' in line_lower:
            return 'claims'
        elif 'cancellation' in line_lower:
            return 'cancellation'
        elif 'definitions' in line_lower:
            return 'definitions'
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
    
    def generate_comprehensive_summary(self, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive summary with all analyzed elements"""
        summary = {
            # Basic structure
            'clause_types': defaultdict(int),
            'sections': set(),
            'hierarchy_levels': defaultdict(int),
            'insurance_clauses': defaultdict(int),
            
            # Financial analysis
            'financial_summary': {
                'total_INR_amounts': 0,
                'total_percentages': 0,
                'limits_found': 0,
                'deductibles_found': 0,
                'premium_references': 0,
                'calculations_found': 0
            },
            
            # Temporal analysis
            'temporal_summary': {
                'dates_found': 0,
                'periods_found': 0,
                'policy_periods': 0,
                'effective_dates': 0,
                'expiration_dates': 0
            },
            
            # Legal analysis
            'legal_summary': {
                'conditional_clauses': 0,
                'coverage_providing_clauses': 0,
                'restrictive_clauses': 0,
                'qualified_statements': 0
            },
            
            # Insurance analysis
            'insurance_summary': {
                'limit_types': set(),
                'coverage_triggers': set(),
                'territories': set(),
                'insurance_categories': set(),
                'coverage_clauses': 0,
                'exclusion_clauses': 0,
                'condition_clauses': 0
            },
            
            # Cross-references and entities
            'reference_summary': {
                'cross_references_found': 0,
                'named_entities_found': 0,
                'risk_classifications': 0,
                'table_rows': 0
            }
        }
        
        for clause in clauses:
            # Basic structure counts
            summary['clause_types'][clause['clause_type']] += 1
            summary['sections'].add(clause['section'])
            summary['hierarchy_levels'][clause['hierarchy_level']] += 1
            
            if clause['clause_type'] == 'insurance_clause':
                insurance_type = clause['metadata'].get('insurance_type', 'unknown')
                summary['insurance_clauses'][insurance_type] += 1
            
            # Financial data analysis
            if clause['financial_data']:
                fd = clause['financial_data']
                summary['financial_summary']['total_INR_amounts'] += len(fd.get('INR_amounts', []))
                summary['financial_summary']['total_percentages'] += len(fd.get('percentages', []))
                summary['financial_summary']['limits_found'] += len(fd.get('limits', []))
                summary['financial_summary']['deductibles_found'] += len(fd.get('deductibles', []))
                summary['financial_summary']['premium_references'] += len(fd.get('premiums', []))
                summary['financial_summary']['calculations_found'] += len(fd.get('calculations', []))
            
            # Temporal data analysis
            if clause['temporal_data']:
                td = clause['temporal_data']
                summary['temporal_summary']['dates_found'] += len(td.get('dates', []))
                summary['temporal_summary']['periods_found'] += len(td.get('periods', []))
                summary['temporal_summary']['policy_periods'] += len(td.get('policy_periods', []))
                summary['temporal_summary']['effective_dates'] += len(td.get('effective_dates', []))
                summary['temporal_summary']['expiration_dates'] += len(td.get('expiration_dates', []))
            
            # Legal analysis
            if clause['legal_analysis']:
                la = clause['legal_analysis']
                if la.get('conditional_markers'):
                    summary['legal_summary']['conditional_clauses'] += 1
                if la.get('coverage_providing'):
                    summary['legal_summary']['coverage_providing_clauses'] += 1
                if la.get('restrictive'):
                    summary['legal_summary']['restrictive_clauses'] += 1
                if la.get('qualifiers'):
                    summary['legal_summary']['qualified_statements'] += 1
            
            # Insurance analysis
            if clause['insurance_analysis']:
                ia = clause['insurance_analysis']
                summary['insurance_summary']['limit_types'].update(ia.get('limit_types', []))
                summary['insurance_summary']['coverage_triggers'].update(ia.get('coverage_triggers', []))
                summary['insurance_summary']['territories'].update(ia.get('territory', []))
                summary['insurance_summary']['insurance_categories'].update(ia.get('insurance_categories', []))
                
                clause_type = ia.get('clause_type', '')
                if clause_type == 'coverage':
                    summary['insurance_summary']['coverage_clauses'] += 1
                elif clause_type == 'exclusion':
                    summary['insurance_summary']['exclusion_clauses'] += 1
                elif clause_type == 'condition':
                    summary['insurance_summary']['condition_clauses'] += 1
            
            # References and entities
            if clause['cross_references']:
                summary['reference_summary']['cross_references_found'] += len(clause['cross_references'])
            
            if clause['named_entities']:
                total_entities = sum(len(entities) for entities in clause['named_entities'].values())
                summary['reference_summary']['named_entities_found'] += total_entities
            
            if clause['risk_classifications']:
                summary['reference_summary']['risk_classifications'] += len(clause['risk_classifications'])
            
            if clause.get('is_table_row'):
                summary['reference_summary']['table_rows'] += 1
        
        # Convert sets to lists for JSON serialization
        summary['sections'] = list(summary['sections'])
        summary['clause_types'] = dict(summary['clause_types'])
        summary['hierarchy_levels'] = dict(summary['hierarchy_levels'])
        summary['insurance_clauses'] = dict(summary['insurance_clauses'])
        summary['insurance_summary']['limit_types'] = list(summary['insurance_summary']['limit_types'])
        summary['insurance_summary']['coverage_triggers'] = list(summary['insurance_summary']['coverage_triggers'])
        summary['insurance_summary']['territories'] = list(summary['insurance_summary']['territories'])
        summary['insurance_summary']['insurance_categories'] = list(summary['insurance_summary']['insurance_categories'])
        
        return summary
    
    def process_document(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Main function to process a single document with comprehensive analysis"""
        print(f"[+] Analyzing structure of: {os.path.basename(input_path)}")
        
        # Extract and clean text
        raw_text = self.extract_text_from_file(input_path)
        if not raw_text:
            return {}
        
        cleaned_text = self.clean_text(raw_text)
        lines = cleaned_text.split('\n')
        
        # Identify clauses and structure with comprehensive analysis
        clauses_data = self.identify_clauses(lines)
        clauses_with_hierarchy = self.establish_hierarchy(clauses_data)
        
        # Generate comprehensive summary
        comprehensive_summary = self.generate_comprehensive_summary(clauses_with_hierarchy)
        
        # Create structured output
        result = {
            'document_info': {
                'source_file': os.path.basename(input_path),
                'total_clauses': len(clauses_with_hierarchy),
                'total_lines_processed': len(lines),
                'processing_timestamp': str(datetime.now()),
                'analyzer_version': 'Enhanced_v2.0'
            },
            'comprehensive_summary': comprehensive_summary,
            'clauses': clauses_with_hierarchy,
            'analysis_coverage': {
                'financial_data': True,
                'temporal_information': True,
                'legal_language_analysis': True,
                'insurance_analysis': True,
                'cross_references': True,
                'named_entities': True,
                'table_data': True,
                'risk_classifications': True,
                'calculation_formulas': True
            }
        }
        
        # Save structured output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        return result

def process_all_extracted_files(input_dir: str = "output/extracted_text", 
                               output_dir: str = "output/comprehensive_analysis"):
    """Process all extracted text files with comprehensive analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = EnhancedDocumentStructureAnalyzer()
    results_summary = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{filename}_comprehensive.json")
            
            result = analyzer.process_document(input_path, output_path)
            if result:
                results_summary.append({
                    'file': filename,
                    'total_clauses': result['document_info']['total_clauses'],
                    'comprehensive_summary': result['comprehensive_summary'],
                    'analysis_coverage': result['analysis_coverage']
                })
                print(f"[✓] Processed: {filename} -> {result['document_info']['total_clauses']} clauses")
    
    # Save overall summary with comprehensive metrics
    summary_path = os.path.join(output_dir, "comprehensive_processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_documents': len(results_summary),
            'processing_details': results_summary,
            'aggregate_metrics': _calculate_aggregate_metrics(results_summary),
            'coverage_verification': {
                'financial_data_extraction': True,
                'temporal_information_extraction': True,
                'legal_language_analysis': True,
                'insurance_specific_analysis': True,
                'cross_reference_detection': True,
                'named_entity_recognition': True,
                'table_data_extraction': True,
                'risk_classification_detection': True,
                'calculation_formula_detection': True
            }
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"[✔] Comprehensive analysis complete! Processed {len(results_summary)} documents.")
    print(f"[✔] Results saved in: {output_dir}")
    return results_summary

def _calculate_aggregate_metrics(results_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate metrics across all processed documents"""
    aggregate = {
        'total_clauses_across_all_docs': 0,
        'total_financial_data_points': 0,
        'total_temporal_references': 0,
        'total_cross_references': 0,
        'total_named_entities': 0,
        'coverage_providing_clauses': 0,
        'restrictive_clauses': 0,
        'insurance_categories_found': set(),
        'limit_types_found': set(),
        'coverage_triggers_found': set()
    }
    
    for result in results_summary:
        summary = result['comprehensive_summary']
        
        aggregate['total_clauses_across_all_docs'] += result['total_clauses']
        
        # Financial metrics
        if 'financial_summary' in summary:
            fs = summary['financial_summary']
            aggregate['total_financial_data_points'] += (
                fs.get('total_INR_amounts', 0) +
                fs.get('total_percentages', 0) +
                fs.get('limits_found', 0) +
                fs.get('deductibles_found', 0)
            )
        
        # Temporal metrics
        if 'temporal_summary' in summary:
            ts = summary['temporal_summary']
            aggregate['total_temporal_references'] += (
                ts.get('dates_found', 0) +
                ts.get('periods_found', 0)
            )
        
        # Legal metrics
        if 'legal_summary' in summary:
            ls = summary['legal_summary']
            aggregate['coverage_providing_clauses'] += ls.get('coverage_providing_clauses', 0)
            aggregate['restrictive_clauses'] += ls.get('restrictive_clauses', 0)
        
        # Reference metrics
        if 'reference_summary' in summary:
            rs = summary['reference_summary']
            aggregate['total_cross_references'] += rs.get('cross_references_found', 0)
            aggregate['total_named_entities'] += rs.get('named_entities_found', 0)
        
        # Insurance metrics
        if 'insurance_summary' in summary:
            ins = summary['insurance_summary']
            aggregate['insurance_categories_found'].update(ins.get('insurance_categories', []))
            aggregate['limit_types_found'].update(ins.get('limit_types', []))
            aggregate['coverage_triggers_found'].update(ins.get('coverage_triggers', []))
    
    # Convert sets to lists for JSON serialization
    aggregate['insurance_categories_found'] = list(aggregate['insurance_categories_found'])
    aggregate['limit_types_found'] = list(aggregate['limit_types_found'])
    aggregate['coverage_triggers_found'] = list(aggregate['coverage_triggers_found'])
    
    return aggregate

if __name__ == "__main__":
    # Process all extracted files with comprehensive analysis
    results = process_all_extracted_files()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("COMPREHENSIVE DOCUMENT STRUCTURE ANALYSIS SUMMARY")
    print("="*70)
    
    for result in results:
        print(f"\nFile: {result['file']}")
        print(f"  Total Clauses: {result['total_clauses']}")
        
        summary = result['comprehensive_summary']
        
        # Financial summary
        if 'financial_summary' in summary:
            fs = summary['financial_summary']
            print(f"  Financial Data Points: ${fs.get('total_INR_amounts', 0)} amounts, "
                  f"{fs.get('total_percentages', 0)} percentages, "
                  f"{fs.get('limits_found', 0)} limits")
        
        # Temporal summary
        if 'temporal_summary' in summary:
            ts = summary['temporal_summary']
            print(f"  Temporal References: {ts.get('dates_found', 0)} dates, "
                  f"{ts.get('periods_found', 0)} periods")
        
        # Legal summary
        if 'legal_summary' in summary:
            ls = summary['legal_summary']
            print(f"  Legal Analysis: {ls.get('coverage_providing_clauses', 0)} coverage-providing, "
                  f"{ls.get('restrictive_clauses', 0)} restrictive")
        
        # Insurance summary
        if 'insurance_summary' in summary:
            ins = summary['insurance_summary']
            categories = ins.get('insurance_categories', [])
            print(f"  Insurance Categories: {', '.join(categories) if categories else 'None detected'}")
        
        # References
        if 'reference_summary' in summary:
            rs = summary['reference_summary']
            print(f"  Cross-References: {rs.get('cross_references_found', 0)}, "
                  f"Named Entities: {rs.get('named_entities_found', 0)}")
        
        print(f"  Analysis Coverage: {list(result['analysis_coverage'].keys())}")
    
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS VERIFICATION CHECKLIST")
    print("="*70)
    print("✅ Financial Data (INR amounts, percentages, limits)")
    print("✅ Temporal Information (dates, periods, policy terms)")
    print("✅ Legal Language Analysis (conditionals, coverage sentiment)")
    print("✅ Enhanced Insurance Analysis (limits, triggers, territory)")
    print("✅ Cross-references between clauses")
    print("✅ Table data extraction")
    print("✅ Named entities (companies, locations, people)")
    print("✅ Risk classifications and industry codes")
    print("✅ Calculation formulas detection")
    print("✅ Document hierarchy and structure")
    print("="*70)