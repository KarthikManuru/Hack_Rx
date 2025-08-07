import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClauseEmbedding:
    """Data class to store clause embeddings with metadata"""
    clause_id: str
    text: str
    embedding: np.ndarray
    section: str
    subsection: str
    clause_type: str
    hierarchy_level: int
    source_file: str
    metadata: Dict[str, Any]

class SemanticEmbeddingSystem:
    """
    Comprehensive semantic embedding system for insurance document clauses
    Uses Sentence-BERT for embeddings and FAISS for fast similarity search
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 embedding_dim: int = 384,
                 faiss_index_type: str = 'flat'):
        """
        Initialize the semantic embedding system
        
        Args:
            model_name: Sentence transformer model name
            embedding_dim: Dimension of embeddings (384 for MiniLM, 768 for others)
            faiss_index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.faiss_index_type = faiss_index_type
        
        # Available models and their dimensions
        self.available_models = {
            'all-MiniLM-L6-v2': 384,          # Fast, good performance
            'all-mpnet-base-v2': 768,         # Best quality
            'all-distilroberta-v1': 768,      # Good for legal text
            'paraphrase-MiniLM-L6-v2': 384,   # Good for similarity
            'multi-qa-MiniLM-L6-cos-v1': 384, # Good for Q&A
        }
        
        # Initialize components
        self.model = None
        self.faiss_index = None
        self.clause_embeddings: List[ClauseEmbedding] = []
        self.id_to_index_map: Dict[str, int] = {}
        
        # Statistics
        self.stats = {
            'total_clauses_embedded': 0,
            'total_documents_processed': 0,
            'embedding_dimension': embedding_dim,
            'model_used': model_name,
            'index_type': faiss_index_type,
            'creation_timestamp': None
        }
        
    def initialize_model(self):
        """Initialize the Sentence-BERT model"""
        logger.info(f"Loading Sentence-BERT model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            actual_dim = self.model.get_sentence_embedding_dimension()
            
            if actual_dim != self.embedding_dim:
                logger.warning(f"Model dimension ({actual_dim}) differs from specified ({self.embedding_dim})")
                self.embedding_dim = actual_dim
                self.stats['embedding_dimension'] = actual_dim
            
            logger.info(f"✓ Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def initialize_faiss_index(self, num_vectors: int = None):
        """Initialize FAISS index for similarity search"""
        logger.info(f"Initializing FAISS index: {self.faiss_index_type}")
        
        if self.faiss_index_type == 'flat':
            # Exact search, best quality
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            
        elif self.faiss_index_type == 'ivf':
            # Approximate search, faster for large datasets
            nlist = min(100, max(1, num_vectors // 10)) if num_vectors else 100
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
        elif self.faiss_index_type == 'hnsw':
            # Hierarchical NSW, good balance of speed and accuracy
            self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.faiss_index.hnsw.efConstruction = 40
            
        else:
            raise ValueError(f"Unsupported index type: {self.faiss_index_type}")
        
        logger.info(f"✓ FAISS index initialized: {type(self.faiss_index).__name__}")
    
    def load_structured_data(self, input_dir: str) -> List[Dict[str, Any]]:
        """Load all structured clause data from JSON files"""
        logger.info(f"Loading structured data from: {input_dir}")
        
        all_clauses = []
        processed_files = 0
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.json') and 'comprehensive' in filename:
                file_path = os.path.join(input_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    source_file = data.get('document_info', {}).get('source_file', filename)
                    clauses = data.get('clauses', [])
                    
                    # Add source file info to each clause
                    for clause in clauses:
                        clause['source_file'] = source_file
                        all_clauses.append(clause)
                    
                    processed_files += 1
                    logger.info(f"✓ Loaded {len(clauses)} clauses from {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")
        
        logger.info(f"✓ Total loaded: {len(all_clauses)} clauses from {processed_files} files")
        self.stats['total_documents_processed'] = processed_files
        
        return all_clauses
    
    def preprocess_text_for_embedding(self, text: str, clause_metadata: Dict[str, Any] = None) -> str:
        """
        Preprocess clause text to optimize for semantic embedding
        
        Args:
            text: Original clause text
            clause_metadata: Additional metadata for context
        
        Returns:
            Preprocessed text optimized for embedding
        """
        # Clean the text
        processed_text = text.strip()
        
        # Add context from metadata if available
        if clause_metadata:
            context_parts = []
            
            # Add clause type context
            clause_type = clause_metadata.get('clause_type', '')
            if clause_type and clause_type != 'paragraph':
                context_parts.append(f"[{clause_type}]")
            
            # Add insurance type context
            if 'insurance_analysis' in clause_metadata:
                insurance_analysis = clause_metadata['insurance_analysis']
                if insurance_analysis.get('clause_type'):
                    context_parts.append(f"[{insurance_analysis['clause_type']}]")
                
                # Add insurance categories
                categories = insurance_analysis.get('insurance_categories', [])
                if categories:
                    context_parts.append(f"[{', '.join(categories)}]")
            
            # Add section context
            section = clause_metadata.get('section', '')
            if section and section != 'PREAMBLE':
                context_parts.append(f"[Section: {section}]")
            
            # Combine context with text
            if context_parts:
                context_prefix = ' '.join(context_parts) + ' '
                processed_text = context_prefix + processed_text
        
        return processed_text
    
    def generate_embeddings(self, clauses: List[Dict[str, Any]], batch_size: int = 32) -> List[ClauseEmbedding]:
        """
        Generate semantic embeddings for all clauses
        
        Args:
            clauses: List of clause dictionaries
            batch_size: Batch size for embedding generation
        
        Returns:
            List of ClauseEmbedding objects
        """
        logger.info(f"Generating embeddings for {len(clauses)} clauses...")
        
        if not self.model:
            self.initialize_model()
        
        clause_embeddings = []
        
        # Prepare texts for embedding
        texts = []
        for clause in clauses:
            preprocessed_text = self.preprocess_text_for_embedding(
                clause['text'], 
                clause
            )
            texts.append(preprocessed_text)
        
        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        
        # Create ClauseEmbedding objects
        for i, (clause, embedding) in enumerate(zip(clauses, embeddings)):
            clause_embedding = ClauseEmbedding(
                clause_id=clause['id'],
                text=clause['text'],
                embedding=embedding,
                section=clause.get('section', ''),
                subsection=clause.get('subsection', ''),
                clause_type=clause.get('clause_type', ''),
                hierarchy_level=clause.get('hierarchy_level', 0),
                source_file=clause.get('source_file', ''),
                metadata=clause
            )
            clause_embeddings.append(clause_embedding)
            self.id_to_index_map[clause['id']] = i
        
        self.clause_embeddings = clause_embeddings
        self.stats['total_clauses_embedded'] = len(clause_embeddings)
        
        logger.info(f"✓ Generated {len(clause_embeddings)} embeddings")
        return clause_embeddings
    
    def build_faiss_index(self, clause_embeddings: List[ClauseEmbedding] = None):
        """Build FAISS index from clause embeddings"""
        if clause_embeddings is None:
            clause_embeddings = self.clause_embeddings
        
        if not clause_embeddings:
            raise ValueError("No clause embeddings available")
        
        logger.info(f"Building FAISS index for {len(clause_embeddings)} embeddings...")
        
        # Initialize index
        self.initialize_faiss_index(len(clause_embeddings))
        
        # Prepare embedding matrix
        embedding_matrix = np.vstack([ce.embedding for ce in clause_embeddings])
        
        # Train index if needed (for IVF)
        if hasattr(self.faiss_index, 'is_trained') and not self.faiss_index.is_trained:
            logger.info("Training FAISS index...")
            self.faiss_index.train(embedding_matrix)
        
        # Add embeddings to index
        self.faiss_index.add(embedding_matrix)
        
        logger.info(f"✓ FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def semantic_search(self, 
                       query: str, 
                       top_k: int = 10,
                       filter_by: Dict[str, Any] = None) -> List[Tuple[ClauseEmbedding, float]]:
        """
        Perform semantic search on the clause database
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_by: Dictionary of filters to apply
        
        Returns:
            List of (ClauseEmbedding, similarity_score) tuples
        """
        if not self.model or not self.faiss_index:
            raise ValueError("Model and index must be initialized")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_k * 2)  # Get more for filtering
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.clause_embeddings):
                clause_embedding = self.clause_embeddings[idx]
                
                # Apply filters if provided
                if filter_by:
                    if not self._apply_filters(clause_embedding, filter_by):
                        continue
                
                results.append((clause_embedding, float(score)))
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def _apply_filters(self, clause_embedding: ClauseEmbedding, filters: Dict[str, Any]) -> bool:
        """Apply filters to clause embeddings"""
        for filter_key, filter_value in filters.items():
            if filter_key == 'clause_type':
                if clause_embedding.clause_type != filter_value:
                    return False
            elif filter_key == 'section':
                if filter_value.lower() not in clause_embedding.section.lower():
                    return False
            elif filter_key == 'source_file':
                if filter_value.lower() not in clause_embedding.source_file.lower():
                    return False
            elif filter_key == 'hierarchy_level':
                if clause_embedding.hierarchy_level != filter_value:
                    return False
            # Add more filter types as needed
        
        return True
    
    def find_similar_clauses(self, 
                           clause_id: str, 
                           top_k: int = 5,
                           exclude_same_section: bool = False) -> List[Tuple[ClauseEmbedding, float]]:
        """Find clauses similar to a given clause"""
        if clause_id not in self.id_to_index_map:
            raise ValueError(f"Clause ID {clause_id} not found")
        
        clause_idx = self.id_to_index_map[clause_id]
        source_clause = self.clause_embeddings[clause_idx]
        
        # Use the clause text as query
        results = self.semantic_search(source_clause.text, top_k + 1)
        
        # Filter out the source clause and optionally same-section clauses
        filtered_results = []
        for clause_embedding, score in results:
            if clause_embedding.clause_id == clause_id:
                continue
            
            if exclude_same_section and clause_embedding.section == source_clause.section:
                continue
            
            filtered_results.append((clause_embedding, score))
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def get_clause_by_id(self, clause_id: str) -> Optional[ClauseEmbedding]:
        """Get a clause embedding by its ID"""
        if clause_id in self.id_to_index_map:
            idx = self.id_to_index_map[clause_id]
            return self.clause_embeddings[idx]
        return None
    
    def save_system(self, output_dir: str):
        """Save the complete embedding system"""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving embedding system to: {output_dir}")
        
        # Save FAISS index
        faiss_path = os.path.join(output_dir, "faiss_index.index")
        faiss.write_index(self.faiss_index, faiss_path)
        
        # Save clause embeddings (without the large embedding arrays)
        clause_data = []
        for ce in self.clause_embeddings:
            clause_data.append({
                'clause_id': ce.clause_id,
                'text': ce.text,
                'section': ce.section,
                'subsection': ce.subsection,
                'clause_type': ce.clause_type,
                'hierarchy_level': ce.hierarchy_level,
                'source_file': ce.source_file,
                'metadata': ce.metadata
            })
        
        embeddings_path = os.path.join(output_dir, "clause_embeddings.json")
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(clause_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Save ID mapping
        mapping_path = os.path.join(output_dir, "id_to_index_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.id_to_index_map, f, indent=2)
        
        # Save system configuration and stats
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'faiss_index_type': self.faiss_index_type,
            'stats': {**self.stats, 'creation_timestamp': str(datetime.now())}
        }
        
        config_path = os.path.join(output_dir, "system_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ System saved successfully")
        logger.info(f"  - FAISS index: {faiss_path}")
        logger.info(f"  - Clause data: {embeddings_path}")
        logger.info(f"  - Configuration: {config_path}")
    
    def load_system(self, input_dir: str):
        """Load a previously saved embedding system"""
        logger.info(f"Loading embedding system from: {input_dir}")
        
        # Load configuration
        config_path = os.path.join(input_dir, "system_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.embedding_dim = config['embedding_dim']
        self.faiss_index_type = config['faiss_index_type']
        self.stats = config['stats']
        
        # Initialize model
        self.initialize_model()
        
        # Load FAISS index
        faiss_path = os.path.join(input_dir, "faiss_index.index")
        self.faiss_index = faiss.read_index(faiss_path)
        
        # Load clause data
        embeddings_path = os.path.join(input_dir, "clause_embeddings.json")
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            clause_data = json.load(f)
        
        # Load ID mapping
        mapping_path = os.path.join(input_dir, "id_to_index_mapping.json")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.id_to_index_map = json.load(f)
        
        # Reconstruct clause embeddings (without the embedding vectors for memory efficiency)
        self.clause_embeddings = []
        for data in clause_data:
            clause_embedding = ClauseEmbedding(
                clause_id=data['clause_id'],
                text=data['text'],
                embedding=np.array([]),  # Empty array - embeddings are in FAISS
                section=data['section'],
                subsection=data['subsection'],
                clause_type=data['clause_type'],
                hierarchy_level=data['hierarchy_level'],
                source_file=data['source_file'],
                metadata=data['metadata']
            )
            self.clause_embeddings.append(clause_embedding)
        
        logger.info(f"✓ System loaded successfully")
        logger.info(f"  - Model: {self.model_name}")
        logger.info(f"  - Embeddings: {len(self.clause_embeddings)}")
        logger.info(f"  - Index size: {self.faiss_index.ntotal}")

def create_embedding_system(input_dir: str = "output/comprehensive_analysis",
                          output_dir: str = "output/semantic_embeddings",
                          model_name: str = "all-MiniLM-L6-v2",
                          faiss_index_type: str = "flat") -> SemanticEmbeddingSystem:
    """
    Create complete semantic embedding system from structured clause data
    
    Args:
        input_dir: Directory containing structured clause JSON files
        output_dir: Directory to save the embedding system
        model_name: Sentence transformer model to use
        faiss_index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
    
    Returns:
        Configured SemanticEmbeddingSystem
    """
    logger.info("="*60)
    logger.info("CREATING SEMANTIC EMBEDDING SYSTEM")
    logger.info("="*60)
    
    # Initialize system
    system = SemanticEmbeddingSystem(
        model_name=model_name,
        faiss_index_type=faiss_index_type
    )
    
    # Load structured data
    clauses = system.load_structured_data(input_dir)
    
    if not clauses:
        raise ValueError("No clause data found in input directory")
    
    # Generate embeddings
    clause_embeddings = system.generate_embeddings(clauses)
    
    # Build FAISS index
    system.build_faiss_index(clause_embeddings)
    
    # Save system
    system.save_system(output_dir)
    
    # Print summary
    logger.info("="*60)
    logger.info("EMBEDDING SYSTEM CREATION COMPLETE")
    logger.info("="*60)
    logger.info(f"📊 Total clauses embedded: {system.stats['total_clauses_embedded']}")
    logger.info(f"📄 Documents processed: {system.stats['total_documents_processed']}")
    logger.info(f"🧠 Model used: {system.stats['model_used']}")
    logger.info(f"📐 Embedding dimension: {system.stats['embedding_dimension']}")
    logger.info(f"🔍 Index type: {system.stats['index_type']}")
    logger.info(f"💾 Saved to: {output_dir}")
    
    return system

def demonstrate_semantic_search(system: SemanticEmbeddingSystem):
    """Demonstrate semantic search capabilities"""
    logger.info("\n" + "="*50)
    logger.info("SEMANTIC SEARCH DEMONSTRATION")
    logger.info("="*50)
    
    # Example queries
    demo_queries = [
        "coverage for property damage",
        "exclusions for cyber attacks",
        "deductible amounts and limits",
        "cancellation policy terms",
        "territory and jurisdiction clauses"
    ]
    
    for query in demo_queries:
        logger.info(f"\n🔍 Query: '{query}'")
        results = system.semantic_search(query, top_k=3)
        
        for i, (clause_embedding, score) in enumerate(results, 1):
            logger.info(f"  {i}. [Score: {score:.3f}] [{clause_embedding.clause_type}]")
            logger.info(f"     Section: {clause_embedding.section}")
            logger.info(f"     Text: {clause_embedding.text[:100]}...")
        
        if not results:
            logger.info("     No results found")

if __name__ == "__main__":
    # Create embedding system
    system = create_embedding_system()
    
    # Demonstrate search capabilities
    demonstrate_semantic_search(system)