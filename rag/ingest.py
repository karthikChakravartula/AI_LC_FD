import sys
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from rank_bm25 import BM25Okapi
import pickle
from lib.common import simple_tokenizer, get_embedding_function


PERSIST_DIR = "./chroma_db_storage"
    
# 1. Force Python to include the root folder in its search path
# Fix the import path for VS Code
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.my_app.supabase_client import fetch_records

def importing_to_vector_database(table) :
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embedding_model = get_embedding_function()
    processed_documents = []
    docIds = []
    metaDataIDS = []
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
            unique_chunk_id = f"{a['id']}_chunk_{j}"
        
            metadata = {
            "source_table": table,
            "source_name": a['name'],
            "supabase_id": a['id'],
            "chunk_index": j
            }
            
            processed_documents.append(chunk)
            metaDataIDS.append(metadata)
            docIds.append(unique_chunk_id)
        

    print(f"Created {len(processed_documents)} total chunks from Supabase data.")

    print("Embedding chunks locally and saving to disk...")
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=embedding_model)
    
    collection.add(
        documents=processed_documents,
        metadatas=metaDataIDS,
        ids=docIds
    )

    print(f"Successfully saved vector database to: {PERSIST_DIR}")

    print(f"collection : {collection.count()}")
    return collection.get();
    
def setBM25Index(processdocs):
    tokenized_corpus = [
        simple_tokenizer(doc) for doc in processdocs.get("documents", [])
    ]
    bm25_index = BM25Okapi(tokenized_corpus)
    payload = {
        "index": bm25_index,
        "documents": processdocs.get("documents", []),
        "tokenized_corpus": tokenized_corpus,
        "metaData": processdocs.get("metadatas", []),
    }
    
    # Save safely
    filepath = os.path.join(PERSIST_DIR, "bm25_index.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(payload, f)
        
    print(f"Index successfully saved to {filepath}")


def ingest():
    processdocs = importing_to_vector_database('accounts')
    setBM25Index(processdocs)
    print("\n\nSUCCESS: Both Chroma and BM25 index are saved and ready!")
    
ingest()
