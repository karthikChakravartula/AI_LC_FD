import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import Collection
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb

# 1. Force Python to include the root folder in its search path
# Fix the import path for VS Code
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.my_app.supabase_client import fetch_records

def importing_to_vector_database(table) :
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    processed_documents = []
    PERSIST_DIR = "./chroma_db_storage"
    COLLECTION_NAME = "langchain"
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, separators=["\n\n","\n",". "," "])
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        client.delete_collection(name=COLLECTION_NAME)
        print("Existing collection cleared successfully.")
    except Exception:
        print("No existing collection found to clear. Starting fresh.")
    for i, a in enumerate(fetch_records(table,"id,name,description,status")):
        text = f"Account: {a['name'] or ''}\nStatus: {a['status'] or ''}\nDescription: {a['description'] or ''}"
        chunks = splitter.split_text(text)
        for j, chunk in enumerate(chunks):
        # Create a LangChain Document object for each chunk
            doc = Document(
                page_content=chunk,
                metadata = {
                    "source_table": table,
                    "source_name": a['name'],
                    "supabase_id": a['id'],
                    "chunk_index": j
                    },
            )
            processed_documents.append(doc)
        

    print(f"Created {len(processed_documents)} total chunks from Supabase data.")

    #PERSIST_DIR = "./chroma_db_storage"

    print("Embedding chunks locally and saving to disk...")
    vector_store = Chroma.from_documents(
        documents=processed_documents,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    print(f"Successfully saved vector database to: {PERSIST_DIR}")

    print(f"collection : {vector_store._collection.count()}")

    # 6. Test Retrieval with Citation Metadata
    query = "which company just closed a funding round"
    results = vector_store.similarity_search(query, k=3)

    print("\n--- Test Query Results ---")
    for doc in results:
        print(f"Matching Text: '{doc.page_content}'")
        print(f"Source Citation -> Table: {doc.metadata['source_table']}, Name : {doc.metadata['source_name']} , Row ID: {doc.metadata['supabase_id']}")




importing_to_vector_database('accounts')