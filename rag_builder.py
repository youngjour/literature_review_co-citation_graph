import os
import PyPDF2

# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI 
# from langchain.embeddings import OpenAIEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io

# Google Drive API credentials
SERVICE_ACCOUNT_FILE = 'google_service_key/confident-coda-260302-dd39ee8cec44.json'
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_google_drive():
    """Authenticate with Google Drive API using a service account."""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=credentials)

def extract_text_from_pdfs(pdf_folder):
    """Extract text from PDF files in the given folder."""
    documents = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(pdf_folder, file_name)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
    return documents

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vector_store(documents, vectorstore_path, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create and save FAISS vector store."""
    from langchain.embeddings import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    vector_store = FAISS.from_texts(texts, embedding=embedding_function, metadatas=metadatas)
    vector_store.save_local(vectorstore_path)
    return vector_store

def main(google_folder_id, vectorstore_path, batch_size=5, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Process PDF files in batches from Google Drive and create vector store."""
    drive_service = authenticate_google_drive()
    query = f"'{google_folder_id}' in parents and mimeType='application/pdf'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    if not files:
        print("No PDF files found in the Google Drive folder.")
        return

    os.makedirs(vectorstore_path, exist_ok=True)
    temp_pdf_folder = "temp_pdfs"
    os.makedirs(temp_pdf_folder, exist_ok=True)

    total_files = len(files)
    for i in range(0, total_files, batch_size):
        batch_files = files[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1}/{(total_files + batch_size - 1) // batch_size}...")

        for file in batch_files:
            file_id = file['id']
            file_name = file['name']
            temp_pdf_path = os.path.join(temp_pdf_folder, file_name)

            vectorstore_file = os.path.join(vectorstore_path, f"{file_name}.faiss")
            metadata_file = os.path.join(vectorstore_path, f"{file_name}.pkl")
            if os.path.exists(vectorstore_file) and os.path.exists(metadata_file):
                print(f"Vector store already exists for {file_name}, skipping embedding.")
                continue

            print(f"Downloading {file_name} from Google Drive...")
            try:
                request = drive_service.files().get_media(fileId=file_id)
                with io.FileIO(temp_pdf_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        print(f"Download progress: {int(status.progress() * 100)}%")
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")
                continue

            try:
                documents = extract_text_from_pdfs(temp_pdf_folder)
                split_docs = split_documents(documents)
                create_vector_store(split_docs, vectorstore_path, embedding_model_name)
                print(f"Vector store saved for {file_name} at {vectorstore_path}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")
                continue

    print("All files processed.")

if __name__ == "__main__":
    GOOGLE_FOLDER_ID = "1rnq1P_IGpzZzXp8iHaKLKt6BlJXkT2FI"
    VECTORSTORE_PATH = "vector_store"
    main(GOOGLE_FOLDER_ID, VECTORSTORE_PATH, batch_size=5)