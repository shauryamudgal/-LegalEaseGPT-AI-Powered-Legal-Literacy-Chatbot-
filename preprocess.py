"""
preprocess.py - Advanced Legal Document Preprocessor for RAG Systems

Features:
1. Multilingual legal text cleaning
2. Smart chunking with legal context preservation
3. Metadata extraction (jurisdiction, document type)
4. Configurable processing pipelines
5. Quality validation layers
"""

import re
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredFileLoader
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_preprocessor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LegalPreprocessor')

@dataclass
class PreprocessorConfig:
    """Configuration for legal document processing"""
    chunk_size: int = 800  # Optimized for legal paragraphs
    chunk_overlap: int = 150  # Maintains context across chunks
    max_text_size: int = 100000
    min_chunk_words: int = 25  # 1MB max per document
    clean_citations: bool = True
    normalize_whitespace: bool = True
    detect_jurisdictions: bool = True
    extract_metadata: bool = True
    languages: List[str] = None  # ['en','hi'] for bilingual

class LegalDocumentPreprocessor:
    """End-to-end preprocessing pipeline for legal documents"""
    
    def __init__(self, config: PreprocessorConfig = None):
        self.config = config or PreprocessorConfig()
        self._compile_patterns()
        self.region_cache = {}
        self.text_splitter = self._create_splitter()

    def _compile_patterns(self):
        """Pre-compile all regex patterns for performance"""
        self.citation_patterns = [
            re.compile(r'\(\d+\s[A-Za-z]+\.\d+\s\d+\)'),  # (123 F.3d 456)
            re.compile(r'\d+\s[A-Za-z]+\.\d+\s\d+'),       # 123 F.3d 456
            re.compile(r'\d+\sU\.?S\.?\s\d+'),             # 123 U.S. 456
            re.compile(r'\[\d+\]\s[A-Z]+\s\d+'),           # [2024] EWCA 123
        ]
        
        self.jurisdiction_patterns = {
            'california': re.compile(r'(California|Cal\.)\s(Court|Code|Statutes?)'),
            'new_york': re.compile(r'(New York|N\.Y\.)\s(Law|Court|CLS)'),
            'india': re.compile(r'(Indian\s(Penal\sCode|Constitution)|IPC\sSection)'),
        }
        
        self.document_type_patterns = {
            'contract': re.compile(r'\b(agreement|contract|lease)\b', re.IGNORECASE),
            'statute': re.compile(r'\b(section|article|statute|act|regulation)\b'),
            'judgment': re.compile(r'\b(v\.?|in re|petitioner|respondent)\b'),
        }

    def _create_splitter(self):
        """Create text splitter with legal-aware separators"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=[
                "\n\n", 
                "\n", 
                "(?<=\\. )",  # Split after sentences
                "(?<=। )",    # Hindi full stop
                "(?<=॥ )",    # Sanskrit double danda
                " ",          # Fallback to words
            ],
            keep_separator=True
        )

    def load_documents(self, input_path: str) -> List[Any]:
        """
        Load documents from file/directory with automatic format detection
        
        Args:
            input_path: File or directory path
            
        Returns:
            List of Langchain Document objects
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Path does not exist: {input_path}")

        if os.path.isfile(input_path):
            return self._load_single_file(input_path)
        return self._load_directory(input_path)

    def _load_single_file(self, file_path: str) -> List[Any]:
        """Load individual file with appropriate loader"""
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
        }
        
        loader_class = loaders.get(ext, UnstructuredFileLoader)
        try:
            return loader_class(file_path).load()
        except Exception as e:
            logger.error(f"Failed loading {file_path}: {str(e)}")
            return []

    def _load_directory(self, dir_path: str) -> List[Any]:
        """Batch load documents from directory"""
        loaders = [
            ('**/*.pdf', PyPDFLoader),
            ('**/*.docx', Docx2txtLoader),
            ('**/*.txt', TextLoader),
        ]
        
        documents = []
        for pattern, loader_class in loaders:
            try:
                loader = DirectoryLoader(
                    dir_path,
                    glob=pattern,
                    loader_cls=loader_class,
                    silent_errors=True
                )
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} {pattern} files")
            except Exception as e:
                logger.warning(f"Failed loading {pattern}: {str(e)}")
        
        return documents

    def clean_text(self, text: str) -> str:
        """Advanced cleaning pipeline for legal text"""
        if len(text) > self.config.max_text_size:
            logger.warning(f"Truncating oversized text ({len(text)} chars)")
            text = text[:self.config.max_text_size]

        # Normalization sequence
        text = self._normalize_whitespace(text)
        if self.config.clean_citations:
            text = self._remove_citations(text)
        text = self._replace_legal_symbols(text)
        text = self._remove_boilerplate(text)
        
        return text.strip()

    def _normalize_whitespace(self, text: str) -> str:
        """Clean and standardize whitespace"""
        text = re.sub(r'\s+', ' ', text)  # Collapse spaces
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit newlines
        return text

    def _remove_citations(self, text: str) -> str:
        """Remove legal citations while preserving context"""
        for pattern in self.citation_patterns:
            text = pattern.sub('', text)
        return text

    def _replace_legal_symbols(self, text: str) -> str:
        """Replace legal symbols with plain text"""
        replacements = {
            '§': 'Section ',
            '¶': 'Paragraph ',
            '©': '(Copyright) ',
            '®': '(Registered) '
        }
        for symbol, replacement in replacements.items():
            text = text.replace(symbol, replacement)
        return text

    def _remove_boilerplate(self, text: str) -> str:
        """Remove common legal boilerplate"""
        patterns = [
            r'Page \d+ of \d+',
            r'CONFIDENTIAL AND PROPRIETARY',
            r'DRAFT(?: FOR REVIEW)?',
            r'Case No\.?\s*[\w-]+'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract legal-specific metadata"""
        metadata = {
            'jurisdiction': self._detect_jurisdiction(text),
            'document_type': self._detect_document_type(text),
            'language': self._detect_language(text)
        }
        return {k: v for k, v in metadata.items() if v}

    def _detect_jurisdiction(self, text: str) -> Optional[str]:
        """Identify legal jurisdiction from text patterns"""
        if not self.config.detect_jurisdictions:
            return None
            
        cached = self.region_cache.get(text[:1000])  # Cache by text fingerprint
        if cached:
            return cached
            
        for region, pattern in self.jurisdiction_patterns.items():
            if pattern.search(text):
                self.region_cache[text[:1000]] = region
                return region
        return 'general'

    def _detect_document_type(self, text: str) -> Optional[str]:
        """Classify legal document type"""
        if not self.config.extract_metadata:
            return None
            
        for doc_type, pattern in self.document_type_patterns.items():
            if pattern.search(text):
                return doc_type
        return 'other'

    def _detect_language(self, text: str) -> Optional[str]:
        """Simple language detection (can be enhanced with langdetect)"""
        if not self.config.languages:
            return None
            
        sample = text[:500]  # Check first 500 characters
        if re.search(r'[\u0900-\u097F]', sample):  # Devanagari chars
            return 'hi'
        elif re.search(r'[\u0980-\u09FF]', sample):  # Bengali
            return 'bn'
        return 'en'  # Default

    def validate_chunk(self, chunk: str) -> bool:
        """Quality check for document chunks"""
        if len(chunk) < 50:
            return False  # Too short
            
        if len(re.findall(r'\w+', chunk)) < 10:
            return False  # Not enough words
            
        if re.fullmatch(r'[\W\d_]+', chunk):
            return False  # No meaningful text
            
        return True

    def process(self, input_path: str) -> List[Any]:
        """
        Complete processing pipeline:
        1. Load -> 2. Clean -> 3. Enrich -> 4. Chunk -> 5. Validate
        """
        logger.info(f"Starting processing: {input_path}")
        
        # 1. Load documents
        documents = self.load_documents(input_path)
        if not documents:
            logger.warning("No documents loaded")
            return []

        processed_chunks = []
        for doc in documents:
            try:
                # 2. Clean text
                doc.page_content = self.clean_text(doc.page_content)
                
                # 3. Extract metadata
                if self.config.extract_metadata:
                    meta = self.extract_metadata(doc.page_content)
                    doc.metadata.update(meta)
                
                # 4. Split into chunks
                chunks = self.text_splitter.split_documents([doc])
                
                # 5. Validate and collect
                for chunk in chunks:
                    if self.validate_chunk(chunk.page_content):
                        processed_chunks.append(chunk)
                        
            except Exception as e:
                logger.error(f"Failed processing document: {str(e)}")
                continue

        logger.info(f"Produced {len(processed_chunks)} valid chunks")
        return processed_chunks

# Example Usage
if __name__ == "__main__":
    config = PreprocessorConfig(
        chunk_size=600,
        languages=['en', 'hi']
    )
    
    processor = LegalDocumentPreprocessor(config)
    chunks = processor.process("/Users/shauryamudgal/Desktop/LegalAidChatbot/data")
    
    # Sample output
    print(f"\nFirst chunk metadata: {chunks[0].metadata}")
    print(f"Content sample: {chunks[0].page_content[:200]}...")