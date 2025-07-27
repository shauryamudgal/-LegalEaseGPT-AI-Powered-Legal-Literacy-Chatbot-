# app.py - Flask backend for legal chatbot (Using Mistral AI)

from dotenv import load_dotenv
load_dotenv()
import os
print(f"DEBUG: ADMIN_TOKEN from environment is: {os.environ.get('ADMIN_TOKEN')}")
import time
import logging
import json
import tempfile
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, render_template, send_from_directory
import whisper
from gtts import gTTS
from langchain_huggingface import HuggingFaceEmbeddings

# --- REMOVED Google GenAI import ---
# import google.generativeai as genai

# Import our custom modules
from preprocess import LegalDocumentPreprocessor
from embed import DocumentEmbedder
# --- Ensure retrieve uses the MODIFIED retrieve.py ---
from retrieve import LegalQueryEngine

# Configure logging (remains the same)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LegalChatbotApp')

# Create Flask app
app = Flask(__name__)

# --- Configuration ---
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", "./vectorstore/indian_lang_index")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
# +++ ADDED Mistral Model Env Var +++
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest") # Default Mistral model
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "default-admin-token-change-me")
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
LOG_FOLDER = "logs"
DATA_FOLDER = "data"

# --- Global variables ---
whisper_model: Optional[Any] = None
query_engine: Optional[LegalQueryEngine] = None
shared_embeddings: Optional[HuggingFaceEmbeddings] = None

# Supported regions (remains the same)
supported_regions: Dict[str, str] = {
    "general": "General / National",
    "delhi": "Delhi",
    "maharashtra": "Maharashtra",
    "karnataka": "Karnataka",
    "tamil_nadu": "Tamil Nadu",
    "west_bengal": "West Bengal",
}

def init_app(app_instance: Flask):
    """Initialize application components and models with Mistral AI."""
    global whisper_model, query_engine, shared_embeddings

    logger.info("--- Initializing Legal Chatbot Application (Mistral AI Backend) ---") # Updated log

    # Create necessary directories (remains the same)
    os.makedirs(LOG_FOLDER, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    os.makedirs(DATA_FOLDER, exist_ok=True)

    app_instance.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Check for essential environment variables
    # --- UPDATED required env var ---
    required_env_vars = ["MISTRAL_API_KEY"] # Check for Mistral key now
    missing_vars = [var for var in required_env_vars if var not in os.environ]

    if missing_vars:
        for var in missing_vars:
            logger.error(f"CRITICAL: Missing environment variable: {var}")
        logger.error("RAG functionality will be disabled.")
        # return # Optionally stop initialization if API key is missing

    # --- REMOVED Google Generative AI configuration block ---

    # Initialize Shared Embeddings Model (remains the same)
    try:
        logger.info(f"Initializing HuggingFace embeddings model: {EMBEDDING_MODEL}")
        shared_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("HuggingFace embeddings initialized successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize HuggingFace embeddings: {str(e)}", exc_info=True)
        shared_embeddings = None

    # Initialize Query Engine (RAG Pipeline) - Modified Condition
    # --- UPDATED condition to check for Mistral key ---
    if shared_embeddings and "MISTRAL_API_KEY" in os.environ:
        try:
            logger.info("Initializing Legal Query Engine with Mistral AI...")
            query_engine = LegalQueryEngine(
                vector_store_path=VECTOR_STORE_PATH,
                embeddings=shared_embeddings,
                model_name=MISTRAL_MODEL # Pass the configured Mistral model
                # Add other Mistral-specific params if needed from LegalQueryEngine.__init__
            )
            success = query_engine.initialize()
            if success:
                logger.info("Legal Query Engine (Mistral) initialized successfully.")
            else:
                logger.error("Failed to initialize Legal Query Engine (Mistral). RAG functionality may be impaired.")
                query_engine = None
        except Exception as e:
            logger.error(f"Failed to initialize Legal Query Engine (Mistral): {str(e)}", exc_info=True)
            query_engine = None
    else:
        logger.warning("Query Engine initialization skipped due to missing embeddings or Mistral API Key.")
        query_engine = None

    # Initialize Whisper model (Speech-to-Text) (remains the same)
    try:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}", exc_info=True)
        logger.warning("Speech recognition will be unavailable.")
        whisper_model = None

    logger.info("--- Application Initialization Complete (Mistral Backend) ---")


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html', regions=supported_regions)

@app.route('/health')
def health():
    """Health endpoint for monitoring."""
    qe_status = "initialized" if query_engine and query_engine.qa_chain else "not initialized"
    embed_status = "initialized" if shared_embeddings else "not initialized"
    whisper_status = "initialized" if whisper_model else "not initialized"
    # --- UPDATED API key check ---
    mistral_api_ok = "MISTRAL_API_KEY" in os.environ

    status = {
        "status": "online",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "components": {
            "llm_provider": "Mistral AI", # Indicate provider
            "embeddings": embed_status,
            "query_engine": qe_status,
            "speech_recognition": whisper_status,
            "mistral_api_configured": mistral_api_ok, # Changed key name
            "vector_store_exists": os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss"))
        }
    }
    # --- UPDATED health condition ---
    is_healthy = bool(shared_embeddings and query_engine and mistral_api_ok)
    return jsonify(status), 200 if is_healthy else 503

# --- handle_text_query remains the same logic, relies on query_engine ---
@app.route('/api/query', methods=['POST'])
def handle_text_query():
    """Handle text-based queries."""
    start_time = time.time()
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON"}), 415

    data = request.get_json()
    query = data.get('query')
    region = data.get('region')
    user_id = data.get('user_id', 'anonymous')

    if not query:
        return jsonify({"success": False, "error": "Missing 'query' parameter"}), 400

    if region and region not in supported_regions:
        logger.warning(f"Received unsupported region: {region}")
        region = "general"

    logger.info(f"Received text query: '{query}', region: {region}, user_id: {user_id}")

    if not query_engine or not query_engine.qa_chain:
        logger.error("Query engine not available or not fully initialized.")
        return jsonify({
            "success": False,
            "error": "Query engine not available",
            "answer": "I'm sorry, the legal information system is currently unavailable. Please try again later.",
            "hindi_answer": "माफ कीजिए, कानूनी सूचना प्रणाली वर्तमान में अनुपलब्ध है। कृपया बाद में पुनः प्रयास करें।"
        }), 503

    try:
        response = query_engine.process_query(query, region, user_id)
        response["query_received"] = query
        response["handler"] = "text"
        response["total_request_time"] = time.time() - start_time
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing text query: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "answer": "I'm sorry, but I encountered an unexpected error while researching your legal question.",
            "hindi_answer": "क्षमा करें, आपके कानूनी प्रश्न पर शोध करते समय मुझे एक अप्रत्याशित त्रुटि का सामना करना पड़ा।"
        }), 500

# --- handle_speech_query remains the same logic, relies on query_engine and whisper ---
@app.route('/api/speech-query', methods=['POST'])
def handle_speech_query():
    """Handle speech-based queries using Whisper for ASR."""
    start_time = time.time()
    if 'audio' not in request.files:
        logger.warning("Speech query attempt with no audio file.")
        return jsonify({"success": False, "error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    region = request.form.get('region', 'general')
    user_id = request.form.get('user_id', 'anonymous')

    if region not in supported_regions:
         logger.warning(f"Received unsupported region via speech form: {region}")
         region = "general"

    logger.info(f"Received speech query: region: {region}, user_id: {user_id}")

    if not whisper_model:
        logger.error("Speech query received but Whisper model is not available.")
        # ... (error response remains same)
        return jsonify({
            "success": False, "error": "Speech recognition not available",
             "answer": "I'm sorry, speech recognition is currently unavailable. Please try text input instead.",
             "hindi_answer": "माफ कीजिए, वाक् पहचान वर्तमान में अनुपलब्ध है। कृपया इसके बजाय टेक्स्ट इनपुट का प्रयास करें."
        }), 503


    if not query_engine or not query_engine.qa_chain:
         logger.error("Speech query received but Query engine not available.")
         # ... (error response remains same)
         return jsonify({
             "success": False, "error": "Query engine not available",
             "answer": "I can understand your speech, but the legal information system is currently unavailable to answer.",
             "hindi_answer": "मैं आपकी बात समझ सकता हूँ, लेकिन कानूनी सूचना प्रणाली वर्तमान में उत्तर देने के लिए अनुपलब्ध है।"
         }), 503

    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=UPLOAD_FOLDER) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        logger.info(f"Transcribing audio file: {temp_audio_path}")
        result = whisper_model.transcribe(temp_audio_path)
        transcription = result["text"].strip()
        detected_language = result.get("language", "unknown")
        logger.info(f"Transcription: '{transcription}', Language: {detected_language}")

        final_response = {
            "success": False, "transcription": transcription,
            "detected_language": detected_language, "handler": "speech",
        }

        if transcription:
            logger.info(f"Processing transcribed query with Mistral: '{transcription}'")
            rag_response = query_engine.process_query(transcription, region, user_id)
            final_response.update(rag_response)
            final_response["success"] = rag_response.get("success", False)

            if final_response["success"] and rag_response.get("answer"):
                try:
                    speech_filename = f"response_{hash(rag_response['answer'])}_{int(time.time())}.mp3"
                    speech_file_path = generate_speech_response(
                        rag_response["answer"], region, speech_filename
                    )
                    final_response["audio_url"] = f"/static/{os.path.basename(speech_file_path)}"
                    logger.info(f"Generated audio response URL: {final_response['audio_url']}")
                except Exception as tts_error:
                    logger.error(f"Failed to generate speech response: {tts_error}", exc_info=True)
                    final_response["tts_error"] = "Failed to generate audio response."
            # ... (rest of speech handling logic remains same)
            elif not final_response["success"]:
                 logger.warning("RAG processing failed for transcribed query.")
            else:
                 logger.warning("RAG processing succeeded but returned no answer text to synthesize.")

        else:
            logger.warning("Transcription resulted in empty text.")
            final_response["error"] = "Could not understand audio or audio was silent."
            final_response["answer"] = "I couldn't understand what you said. Could you please try again?"
            final_response["hindi_answer"] = "मैं समझ नहीं पाया कि आपने क्या कहा। क्या आप कृपया पुनः प्रयास कर सकते हैं?"


        final_response["total_request_time"] = time.time() - start_time
        return jsonify(final_response)

    except Exception as e:
        logger.error(f"Error processing speech query: {str(e)}", exc_info=True)
        # ... (error response remains same)
        return jsonify({
            "success": False, "error": f"Internal server error during speech processing: {str(e)}",
            "answer": "I'm sorry, but I encountered an unexpected error processing your spoken query.",
             "hindi_answer": "क्षमा करें, आपके बोले गए प्रश्न को संसाधित करते समय मुझे एक अप्रत्याशित त्रुटि का सामना करना पड़ा।"
        }), 500

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.debug(f"Removed temporary audio file: {temp_audio_path}")
            except Exception as e_clean:
                logger.error(f"Error removing temporary audio file {temp_audio_path}: {str(e_clean)}")


# --- generate_speech_response remains the same, uses gTTS ---
def generate_speech_response(text: str, region: Optional[str], filename: str) -> str:
    """Generate speech response using gTTS."""
    if not text:
        raise ValueError("Cannot generate speech from empty text.")
    output_path = os.path.join(STATIC_FOLDER, filename)
    processed_text = text
    if region and query_engine:
        try:
            processed_text = query_engine.apply_regional_dialect(text, region)
            if processed_text != text:
                 logger.debug(f"Applied dialect adjustment for TTS (region: {region})")
        except Exception as dialect_err:
             logger.warning(f"Could not apply regional dialect for region {region}: {dialect_err}")

    lang = "en"
    tld = "co.in"
    if region in ["california", "new_york", "texas", "florida"]:
         tld = "com"

    logger.info(f"Generating speech with gTTS: lang='{lang}', tld='{tld}'")
    try:
        tts = gTTS(text=processed_text, lang=lang, tld=tld, slow=False)
        tts.save(output_path)
        logger.info(f"Generated speech response saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"gTTS failed to generate speech: {str(e)}", exc_info=True)
        raise

# --- serve_static remains the same ---
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve files from the static directory."""
    return send_from_directory(STATIC_FOLDER, filename)


# --- init_vectorstore_endpoint remains the same logic, relies on Embedder and QueryEngine re-initialization ---
@app.route('/api/init-vectorstore', methods=['POST'])
def init_vectorstore_endpoint():
    """Endpoint to initialize or update the vector store."""
    auth_header = request.headers.get('Authorization')
    token = None
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]

    if not token or token != ADMIN_TOKEN:
        logger.warning("Unauthorized attempt to access /api/init-vectorstore")
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    data = request.get_json() if request.is_json else {}
    data_dir = data.get('data_dir', DATA_FOLDER)

    if not os.path.isdir(data_dir):
         logger.error(f"Provided data directory does not exist or is not a directory: {data_dir}")
         return jsonify({"success": False, "error": f"Data directory not found: {data_dir}"}), 400

    logger.info(f"Received request to initialize vector store from directory: {data_dir}")

    global query_engine, shared_embeddings # Allow modification

    try:
        logger.info("Initializing LegalDocumentPreprocessor...")
        preprocessor = LegalDocumentPreprocessor()
        logger.info(f"Processing documents from: {data_dir}")
        documents = preprocessor.process(data_dir)

        if not documents:
            logger.warning(f"No valid documents found or processed in {data_dir}.")
            return jsonify({"success": False, "message": "No documents found or processed", "processed_count": 0}), 400

        logger.info(f"Successfully processed {len(documents)} document chunks.")

        if not shared_embeddings:
             logger.error("Embeddings model not initialized. Cannot create vector store.")
             return jsonify({"success": False, "error": "Embeddings model failed to initialize earlier."}), 500

        logger.info(f"Initializing DocumentEmbedder with path: {VECTOR_STORE_PATH}")
        embedder = DocumentEmbedder(
            vector_store_path=VECTOR_STORE_PATH,
            model_name=EMBEDDING_MODEL
        )

        logger.info("Creating/Updating FAISS vector store...")
        vector_store = embedder.create_vector_store(documents)
        embedder.save_vector_store(vector_store)
        stats = embedder.get_index_stats()
        logger.info(f"Vector store saved. Stats: {stats}")

        # --- Re-initialize Query Engine (using Mistral configuration) ---
        logger.info("Attempting to re-initialize the global Legal Query Engine (Mistral) with the new vector store...")
        # +++ Ensure MISTRAL_API_KEY exists before trying to re-init +++
        if "MISTRAL_API_KEY" in os.environ:
            try:
                query_engine = LegalQueryEngine(
                    vector_store_path=VECTOR_STORE_PATH,
                    embeddings=shared_embeddings,
                    model_name=MISTRAL_MODEL # Use configured Mistral model
                )
                success = query_engine.initialize()
                if success:
                    logger.info("Global Legal Query Engine (Mistral) re-initialized successfully.")
                    message = f"Successfully processed {len(documents)} chunks, created vector store, and re-initialized query engine (Mistral)."
                    status_code = 200
                else:
                    logger.error("Vector store created, but failed to re-initialize the query engine (Mistral).")
                    query_engine = None
                    message = f"Processed {len(documents)} chunks and created vector store, but failed to re-initialize query engine (Mistral)."
                    status_code = 207
            except Exception as qe_reinit_error:
                 logger.error(f"Error re-initializing query engine (Mistral) after vector store update: {qe_reinit_error}", exc_info=True)
                 query_engine = None
                 message = f"Processed {len(documents)} chunks and created vector store, but encountered an error re-initializing query engine (Mistral): {qe_reinit_error}"
                 status_code = 500
        else:
             # If API key is missing, vector store is updated but engine cannot be re-initialized
             logger.warning("Mistral API key missing. Vector store updated, but Query Engine cannot be re-initialized.")
             query_engine = None # Ensure it's None
             message = f"Successfully processed {len(documents)} chunks and updated vector store. Query engine NOT re-initialized (Missing MISTRAL_API_KEY)."
             status_code = 207 # Indicate partial success

        return jsonify({
            "success": status_code == 200,
            "message": message,
            "processed_count": len(documents),
            "vector_store_stats": stats
        }), status_code

    except Exception as e:
        logger.error(f"Error during vector store initialization endpoint: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500


# --- Error handlers remain the same ---
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 Not Found: {request.path}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 Internal Server Error: {e}", exc_info=True)
    return render_template('500.html', error=e), 500


# Initialize app components when this module is loaded
init_app(app)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    logger.info(f"Starting Flask server (Mistral Backend) on port {port} with debug mode: {debug}")
    app.run(ssl_context=('cert.pem', 'key.pem'),host='0.0.0.0', port=port, debug=debug)