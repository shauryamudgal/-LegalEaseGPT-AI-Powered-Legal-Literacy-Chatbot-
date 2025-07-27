"""
embed.py - Stable Embedding Pipeline for Indian Legal RAG System (macOS + Python 3.12 safe)
"""

import os
import logging
import shutil
import gc
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from tqdm import tqdm
import faiss

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_embedding.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('IndianLegalEmbedder')
faiss.verbose = False

class SafeSentenceEmbedder(Embeddings):
    """Custom embedding class avoiding multiprocessing for stability"""
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info("Embedding documents one by one to avoid parallelism")
        return [self.model.encode(text, normalize_embeddings=True) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True)

class DocumentEmbedder:
    def __init__(self,
                 vector_store_path: str = "./vectorstore/indian_lang_index"):
        self.vector_store_path = vector_store_path
        self.embeddings = SafeSentenceEmbedder()
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        logger.info(f"Creating vector store with {len(documents)} documents")
        try:
            vector_store = FAISS.from_documents(
                tqdm(documents, desc="Embedding documents"),
                self.embeddings
            )
            logger.info("Vector store created successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise

    def save_vector_store(self, vector_store: FAISS) -> None:
        try:
            temp_path = f"{self.vector_store_path}_temp"
            final_path = self.vector_store_path
            vector_store.save_local(temp_path)

            if os.path.exists(final_path):
                shutil.rmtree(final_path)

            os.rename(temp_path, final_path)

            os.chmod(final_path, 0o755)
            for root, dirs, files in os.walk(final_path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o755)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o644)

            logger.info(f"Saved vector store to {final_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise

    def load_vector_store(self) -> FAISS:
        try:
            if not os.path.exists(f"{self.vector_store_path}/index.faiss"):
                raise FileNotFoundError("Vector store not found")

            vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded vector store from {self.vector_store_path}")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise

    def update_vector_store(self, new_documents: List[Document]) -> FAISS:
        try:
            vector_store = self.load_vector_store()
            logger.info(f"Found existing vector store with {vector_store.index.ntotal} vectors")
        except:
            logger.info("No existing vector store found, creating new one")
            return self.create_vector_store(new_documents)

        logger.info(f"Adding {len(new_documents)} new documents")
        vector_store.add_documents(new_documents)
        return vector_store

    def get_index_stats(self) -> Dict[str, Any]:
        try:
            vector_store = self.load_vector_store()
            return {
                "vectors": vector_store.index.ntotal,
                "dimensions": vector_store.index.d,
                "type": "FAISS",
                "path": self.vector_store_path
            }
        except Exception as e:
            logger.error(f"Could not get index stats: {str(e)}")
            return {}

if __name__ == "__main__":
    from preprocess import LegalDocumentPreprocessor

    preprocessor = LegalDocumentPreprocessor()
    embedder = DocumentEmbedder()

    try:
        documents = preprocessor.process("./data/")
        if documents:
            vector_store = embedder.create_vector_store(documents)
            embedder.save_vector_store(vector_store)

            stats = embedder.get_index_stats()
            print(f"\nVector Store Stats:")
            print(f"- Documents: {stats.get('vectors', 0)}")
            print(f"- Dimensions: {stats.get('dimensions', 0)}")
        else:
            logger.error("No documents processed")
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
    finally:
        gc.collect()
