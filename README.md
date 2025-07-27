🧠 LegalEaseGPT – AI-Powered Legal Chatbot (RAG + Regional TTS) 🇮🇳
LegalEaseGPT is an AI chatbot designed to simplify Indian legal information like FIR filing, RTI, tenant rights, and labor laws. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide relevant, accurate answers and even converts them to regional-language audio responses using gTTS.

🚀 Features
✅ Ask legal questions and get simplified, LLM-powered answers

📚 Retrieves relevant legal content from embedded documents

🔊 Converts responses to audio in regional languages (Hindi, Tamil, Telugu, etc.)

🌐 Flask API for embedding, querying, and audio delivery

🔐 .env protected with support for multiple LLM providers (OpenAI, Mistral)

⚙️ How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/LegalEaseGPT.git
cd LegalEaseGPT

2. Install Requirements
pip install -r requirements.txt

3. Set Up Environment Variables
Create a .env file in the root:

cp .env.example .env

Then fill in your actual keys:

OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
MISTRAL_API_KEY=your-mistral-api-key
ADMIN_TOKEN=your-admin-token

4. Ingest Legal Documents
Add your legal PDFs or text files to the uploads/ folder.

Then embed them into the vectorstore using:

python embed.py

5. Start the Flask API Server
python app.py

The API will run at http://127.0.0.1:5000

6. Make a Query
You can use curl or Postman:

curl -X POST http://127.0.0.1:5000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "How to file an RTI?", "region": "hi"}'

The response will include:

A generated answer

A path to the audio file with spoken response (in Hindi here)

📂 Project Structure and File Order
Understanding the role and order of the main Python files is crucial for setting up and extending LegalEaseGPT:

preprocess.py:

Role: Defines the LegalDocumentPreprocessor class, which is responsible for loading raw legal documents (PDFs, DOCX, TXT), cleaning their content, and splitting them into manageable, context-rich chunks. This is the first step in preparing your data.

embed.py:

Role: Contains the DocumentEmbedder class. It takes the preprocessed document chunks (from preprocess.py), converts them into numerical vector embeddings using a Sentence Transformer model, and stores these embeddings in a FAISS vector store for efficient retrieval.

retrieve.py:

Role: Implements the LegalQueryEngine. This is the core of the Retrieval-Augmented Generation (RAG) pipeline. It uses the vector store (created by embed.py) to find relevant document chunks for a given query and then leverages a Large Language Model (Mistral AI in this case) to generate a coherent and accurate answer based on the retrieved context.

inspect_chunks.py:

Role: A utility script for development and debugging. It uses preprocess.py to load and process documents and then prints the resulting chunks and their metadata. This helps verify that your preprocessing step is working as expected before embedding.

app.py:

Role: The main Flask application that orchestrates all the above components. It exposes API endpoints for querying (text and speech), managing the vector store, and serves the frontend. This is the entry point for running the complete LegalEaseGPT system.

🗣️ TTS Language Support
LegalEaseGPT uses gTTS to support:

hi – Hindi

ta – Tamil

te – Telugu

(More languages can be added easily)

🔐 Security
Ensure your .env file is excluded via .gitignore

Never commit or expose API keys

Use ADMIN_TOKEN to protect any admin-only API routes

✅ Future Improvements
Add voice input via Whisper/Vosk

Expand legal topics and regional language support

Add frontend UI with Streamlit or React

Offline LLM and TTS support for rural 