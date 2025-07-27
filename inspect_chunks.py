import os
import logging
from preprocess import LegalDocumentPreprocessor # Import your preprocessor class

# Configure basic logging (optional, but helpful)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Use the same data folder your main app uses for indexing
DATA_FOLDER = os.environ.get("DATA_FOLDER", "./data")
# You could potentially load other preprocessor config here if needed

def inspect_document_chunks(data_dir: str):
    """
    Loads documents, processes them into chunks using LegalDocumentPreprocessor,
    and prints the content and metadata of each chunk.
    """
    if not os.path.isdir(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        return

    logging.info(f"Initializing LegalDocumentPreprocessor...")
    # Initialize with default config, or load specific config if your app does
    preprocessor = LegalDocumentPreprocessor()

    logging.info(f"Processing documents from: {data_dir}")
    # Use the process method which returns a list of LangChain Document objects
    documents = preprocessor.process(data_dir)

    if not documents:
        logging.warning(f"No documents were processed from {data_dir}.")
        return

    print(f"\n--- Found {len(documents)} Chunks ---")

    for i, doc in enumerate(documents):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content}")
        print("-" * 20) # Separator

    logging.info(f"Finished inspecting {len(documents)} chunks.")

if __name__ == "__main__":
    inspect_document_chunks(DATA_FOLDER)