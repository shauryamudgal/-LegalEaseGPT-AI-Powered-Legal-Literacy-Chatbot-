# retrieve.py - Indian Legal RAG Query Engine (Using Mistral AI)

import os
import re
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv # Added for standalone testing
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# --- REMOVED Google GenAI imports ---
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
# +++ ADDED Mistral AI import +++
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings # Ensure this is available
from langchain.prompts import PromptTemplate

# Load .env file for standalone testing
load_dotenv()

# Configure logging (remains the same)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('IndianLegalRetriever')

# Default Prompt (remains the same, should work well with Mistral)
DEFAULT_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. Provide a concise and legally accurate answer based ONLY on the provided context. If you don't know the answer from the context, just say that you don't know, don't try to make up an answer. Mention relevant sections or case names if found in the context.

Context:
{context}

Question: {question}

Helpful Answer (in English):
"""

class LegalQueryEngine:
    """Query engine specialized for Indian legal system, using Mistral AI"""

    def __init__(self,
                 vector_store_path: str = "./vectorstore/indian_lang_index",
                 embeddings: Any = None, # MUST be provided by the caller (app.py)
                 # +++ UPDATED model name for Mistral +++
                 model_name: str = "mistral-small-latest", # Default to mistral-small
                 temperature: float = 0.6,
                 top_k: int = 4,
                 score_threshold: float = 0.4,
                 prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
                 max_tokens: int = 1024): # Added max_tokens for Mistral
        """
        Initialize the query engine for Indian legal system with Mistral AI.

        Args:
            vector_store_path: Path to FAISS vectorstore
            embeddings: Embeddings model instance (must match index creation)
            model_name: Mistral model name (e.g., "mistral-large-latest", "mistral-small-latest", "open-mixtral-8x7b")
            temperature: Creativity control (0-1)
            top_k: Number of documents to retrieve
            score_threshold: Minimum relevance score for retrieved documents
            prompt_template: Template string for the QA prompt
            max_tokens: Maximum tokens for the LLM response
        """
        if embeddings is None:
             raise ValueError("Embeddings model instance must be provided during initialization.")

        self.vector_store_path = vector_store_path
        self.embeddings = embeddings
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.prompt_template_str = prompt_template
        self.max_tokens = max_tokens # Store max_tokens

        # Initialize components to None
        self.vector_store: Optional[FAISS] = None
        self.llm: Optional[ChatMistralAI] = None # Changed type hint
        self.qa_chain: Optional[RetrievalQA] = None

        # India-specific configuration (remains the same)
        self.indian_states = {
            'maharashtra': ['mumbai', 'pune', 'maharashtra', 'bombay high court'],
            'delhi': ['delhi', 'nct', 'delhi high court', 'tis hazari'],
            'karnataka': ['bangalore', 'bengaluru', 'karnataka', 'high court of karnataka'],
            'tamil_nadu': ['chennai', 'tamil nadu', 'madras high court'],
            'west_bengal': ['kolkata', 'calcutta', 'west bengal', 'calcutta high court'],
        }
        self.legal_terms = {
            'ipc': 'Indian Penal Code',
            'cpc': 'Code of Civil Procedure',
            'crpc': 'Code of Criminal Procedure',
            'constitution': 'Constitution of India'
        }

    def initialize(self) -> bool:
        """Initialize the query engine components with Mistral AI"""
        logger.info("Initializing LegalQueryEngine with Mistral AI...")
        try:
            # 1. Load vector store (remains the same)
            if not os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
                 logger.error(f"Vector store index file not found at: {os.path.join(self.vector_store_path, 'index.faiss')}")
                 return False

            logger.info(f"Loading vector store from {self.vector_store_path} using provided embeddings.")
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
                index_name="index"
            )
            logger.info(f"Vector store loaded successfully with {self.vector_store.index.ntotal} vectors.")

            # 2. Verify Mistral API key
            # --- REMOVED Google API key check ---
            # +++ ADDED Mistral API key check +++
            if "MISTRAL_API_KEY" not in os.environ:
                logger.error("MISTRAL_API_KEY not found in environment variables. LLM will not function.")
                # Allow initialization to continue, but log error. Querying will fail later.
                # return False # Optionally return False here to prevent incomplete initialization
            else:
                logger.info("MISTRAL_API_KEY found.")

            # --- REMOVED genai.configure ---

            # 3. Initialize LLM (Changed to ChatMistralAI)
            logger.info(f"Initializing Mistral LLM: {self.model_name}")
            # +++ Initialized ChatMistralAI +++
            # The API key is automatically picked up from MISTRAL_API_KEY env var by default
            self.llm = ChatMistralAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # mistral_api_key=os.environ.get("MISTRAL_API_KEY") # Can be passed explicitly if needed
            )
            logger.info("Mistral LLM initialized.")

            # 4. Create retrieval chain (remains largely the same)
            logger.info("Creating RetrievalQA chain.")
            retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.top_k,
                    "score_threshold": self.score_threshold
                }
            )

            # Create prompt template (remains the same)
            QA_PROMPT = PromptTemplate(
                template=self.prompt_template_str,
                input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, # Use the initialized Mistral LLM
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_PROMPT}
            )
            logger.info("RetrievalQA chain created successfully.")
            logger.info("LegalQueryEngine initialized successfully with Mistral AI.")
            return True

        except FileNotFoundError as fnf_error:
            logger.error(f"Initialization failed: Vector store not found at {self.vector_store_path}. Error: {str(fnf_error)}")
            return False
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            self.vector_store = None
            self.llm = None
            self.qa_chain = None
            return False

    # detect_indian_jurisdiction, expand_legal_terms remain the same
    def detect_indian_jurisdiction(self, query: str) -> Optional[str]:
        """Detect Indian state jurisdiction from query text."""
        query_lower = query.lower()
        for state, keywords in self.indian_states.items():
            if any(keyword in query_lower for keyword in keywords):
                logger.debug(f"Detected jurisdiction: {state}")
                return state
        if any(term in query_lower for term in ['supreme court', 'sc', 'india', 'national']):
            logger.debug("Detected jurisdiction: national")
            return 'national'
        logger.debug("No specific Indian jurisdiction detected.")
        return None

    def expand_legal_terms(self, query: str) -> str:
        """Expand abbreviated Indian legal terms."""
        original_query = query
        for short, full in self.legal_terms.items():
            query = re.sub(rf'\b{short}\b', full, query, flags=re.IGNORECASE)
        if query != original_query:
             logger.debug(f"Expanded legal terms in query: '{original_query}' -> '{query}'")
        return query


    def process_query(self, query: str, region: Optional[str] = None, user_id: Optional[str] = None) -> Dict:
        """Process a legal query using Mistral AI."""
        start_time = time.time()
        log_data = {
            "query": query,
            "region": region,
            "user_id": user_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "llm_provider": "Mistral AI", # Added provider info
            "llm_model": self.model_name
        }

        if not self.qa_chain or not self.llm: # Added check for LLM too
            logger.error("Query engine or LLM not initialized. Cannot process query.")
            if not self.initialize(): # Attempt re-init
                 log_data["error"] = "System initialization failed"
                 log_data["processing_time"] = time.time() - start_time
                 self.log_query(log_data)
                 return self._error_response("System initialization failed")
            else:
                 logger.info("Re-initialized query engine successfully.")


        # --- Added check for API key before proceeding ---
        if "MISTRAL_API_KEY" not in os.environ:
            logger.error("MISTRAL_API_KEY missing. Cannot execute query.")
            log_data["error"] = "MISTRAL_API_KEY missing"
            log_data["processing_time"] = time.time() - start_time
            self.log_query(log_data)
            return self._error_response("LLM API key not configured")

        try:
            # Pre-process query (remains the same)
            detected_jurisdiction = self.detect_indian_jurisdiction(query)
            log_data["detected_jurisdiction"] = detected_jurisdiction
            expanded_query = self.expand_legal_terms(query)
            log_data["expanded_query"] = expanded_query

            logger.info(f"Processing expanded query with Mistral: '{expanded_query}'")

            # Execute retrieval and generation (remains the same logic)
            if hasattr(self.qa_chain, 'invoke'):
                result = self.qa_chain.invoke({"query": expanded_query})
            else:
                result = self.qa_chain({"query": expanded_query})

            answer = result.get("result", "").strip()
            sources = result.get("source_documents", [])

            # Handle empty answers (remains the same logic)
            if not answer and not sources:
                 logger.warning("Mistral LLM returned empty answer and no sources found.")
                 answer = "I couldn't find specific information related to your query in the available legal documents."
            elif not answer and sources:
                 logger.warning("Mistral LLM returned empty answer, but sources were found. Using generic response.")
                 answer = "I found some potentially relevant documents, but could not synthesize a direct answer. Please review the sources."


            # Post-process answer (remains the same)
            formatted_answer = self._format_indian_citations(answer)

            # Placeholder for Hindi Translation (remains the same)
            hindi_answer = None

            # Log success
            log_data["success"] = True
            log_data["answer"] = formatted_answer
            log_data["hindi_answer_generated"] = bool(hindi_answer)
            log_data["num_sources"] = len(sources)
            log_data["processing_time"] = time.time() - start_time
            self.log_query(log_data)

            # Prepare response (remains the same structure)
            return {
                "success": True,
                "answer": formatted_answer,
                "hindi_answer": hindi_answer,
                "jurisdiction": detected_jurisdiction,
                "sources": [self._format_source(doc) for doc in sources],
                "processing_time": log_data["processing_time"]
            }

        except Exception as e:
            # Log specific Mistral errors if possible, otherwise generic
            logger.error(f"Mistral query processing error for query '{query}': {str(e)}", exc_info=True)
            log_data["success"] = False
            log_data["error"] = str(e)
            log_data["processing_time"] = time.time() - start_time
            self.log_query(log_data)
            # Check if it's an authentication error
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                 return self._error_response("LLM authentication failed. Please check the API key.")
            else:
                 return self._error_response(f"An error occurred: {str(e)}")

    # _format_indian_citations, _format_source, _error_response, apply_regional_dialect, log_query remain the same
    def _format_indian_citations(self, text: str) -> str:
        """Format common Indian legal citation patterns."""
        try:
             text = re.sub(r'(?:Section|Sec\.?)\s*(\d+)\s*(?:of)?\s*(?:the)?\s*IPC', r'Indian Penal Code Section \1', text, flags=re.IGNORECASE)
             text = re.sub(r'\bIPC\s+(\d+)', r'Indian Penal Code Section \1', text)
             text = re.sub(r'(?:Article|Art\.?)\s*(\d+)\s*(?:of)?\s*(?:the)?\s*Constitution', r'Article \1 of the Constitution of India', text, flags=re.IGNORECASE)
             text = re.sub(r'(?:Section|Sec\.?)\s*(\d+)\s*(?:of)?\s*(?:the)?\s*CrPC', r'Code of Criminal Procedure Section \1', text, flags=re.IGNORECASE)
             text = re.sub(r'\bCrPC\s+(\d+)', r'Code of Criminal Procedure Section \1', text)
             text = re.sub(r'(?:Section|Sec\.?)\s*(\d+)\s*(?:of)?\s*(?:the)?\s*CPC', r'Code of Civil Procedure Section \1', text, flags=re.IGNORECASE)
             text = re.sub(r'\bCPC\s+(\d+)', r'Code of Civil Procedure Section \1', text)
             text = re.sub(
                 r'(\b[A-Za-z\s\.]+)\s*(?:v\.?|vs\.?)\s*([A-Za-z\s\.]+)\s*\(?(\d{4})\)?\s*(\d*\s*[A-Z]+\.?\s*\d+)',
                 r'\1 v. \2 (\3) \4',
                 text
             )
        except Exception as e:
             logger.warning(f"Error during citation formatting: {e}")
             return text
        return text

    def _format_source(self, document) -> Dict:
        """Format source document for response."""
        metadata = document.metadata or {}
        content_preview = document.page_content[:300].strip() + ("..." if len(document.page_content) > 300 else "")
        return {
            "content": content_preview,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", None),
            "jurisdiction": metadata.get("jurisdiction"),
            "document_type": metadata.get("document_type"),
        }

    def _error_response(self, error_msg: str) -> Dict:
        """Standard error response format"""
        return {
            "success": False,
            "error": error_msg,
            "answer": f"Sorry, I encountered an error: {error_msg}. Please try again or rephrase your question.",
            "hindi_answer": f"माफ कीजिए, मुझे एक त्रुटि का सामना करना पड़ा: {error_msg}। कृपया पुनः प्रयास करें या अपना प्रश्न बदलें।",
            "sources": [],
            "processing_time": 0
        }

    def apply_regional_dialect(self, text: str, region: Optional[str]) -> str:
         """Apply regional dialect adjustments (basic example)."""
         if not region or region == "general" or region == "national":
            return text
         logger.debug(f"Applying dialect adjustments for region: {region}")
         if region == "california":
            text = text.replace("attorney", "lawyer")
            text = text.replace("petitioner", "plaintiff")
         elif region == "new_york":
            text = text.replace("shall", "must")
            text = text.replace("therein", "in it")
         return text

    def log_query(self, query_data: Dict) -> None:
        """Log query details for analytics"""
        try:
            log_dir = "./logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "query_log.jsonl")
            with open(log_file, "a", encoding="utf-8") as f:
                serializable_data = json.dumps(query_data, ensure_ascii=False, default=str)
                f.write(serializable_data + "\n")
        except Exception as e:
            logger.error(f"Failed to log query: {str(e)}")

# Example usage (updated for Mistral)
if __name__ == "__main__":
    # Requires MISTRAL_API_KEY in environment
    if "MISTRAL_API_KEY" not in os.environ:
        print("Please set the MISTRAL_API_KEY environment variable.")
    else:
        try:
            print("Initializing embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("Embeddings initialized.")

            print("Initializing query engine with Mistral AI...")
            # +++ Specify Mistral model for testing if desired +++
            engine = LegalQueryEngine(
                embeddings=embeddings,
                model_name="mistral-small-latest" # Or another Mistral model
            )
            if engine.initialize():
                 print("Query engine initialized successfully.")
                 query = "What is the punishment under IPC 302?"
                 print(f"\nTesting query: {query}")
                 response = engine.process_query(query)

                 print(f"\nQuery: {query}")
                 print(f"Success: {response.get('success')}")
                 if response.get('success'):
                     print(f"Answer: {response.get('answer')}")
                     print(f"Jurisdiction: {response.get('jurisdiction')}")
                     print("\nSources:")
                     if response.get('sources'):
                         for i, source in enumerate(response["sources"], 1):
                             print(f"{i}. Source: {source.get('source', 'N/A')}, Page: {source.get('page', 'N/A')}")
                     else:
                         print("No sources provided.")
                 else:
                      print(f"Error: {response.get('error')}")
                      print(f"Fallback Answer (Eng): {response.get('answer')}")
                      print(f"Fallback Answer (Hin): {response.get('hindi_answer')}")

            else:
                 print("Failed to initialize query engine.")

        except Exception as main_e:
            print(f"An error occurred during the example run: {main_e}")
            import traceback
            traceback.print_exc()