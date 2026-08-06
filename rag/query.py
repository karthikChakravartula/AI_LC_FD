import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai
import os
from dotenv import load_dotenv 
from lib.common import simple_tokenizer, get_embedding_function
import pickle
import numpy as np

PERSIST_DIR = "./chroma_db_storage"

def get_Apikey():
    load_dotenv()#dotenv_path=current_dir / ".env")
    return os.getenv("GEMINI_API_KEY")

def retreiveChromaDb(question , k = 5) :
   client = chromadb.PersistentClient(path=PERSIST_DIR)
   collection = client.get_collection("langchain", embedding_function=get_embedding_function())
   results = collection.query(query_texts=[question], n_results=k)
   res_list = [{
       "id" : results["ids"][0][idx],
       "document" : results["documents"][0][idx],
       "metadata" : results["metadatas"][0][idx],
   } for idx in range(len(results["ids"][0]))]
   print("From Chroma")
   print(res_list)
   return res_list

def createPromptAndCallLLM(ReciprocalRank :list[str], chromaList : list[str] , bm25List : list[str], question):

    formatted_items = []
    all_docs = {d["metadata"]["supabase_id"]: d for d in chromaList + bm25List}

    for doc_id, score in ReciprocalRank[:5]:
            doc = all_docs.get(doc_id)
            if doc:
                doc_strings = [f"Content : {doc['document']}, Id : {doc['metadata']['supabase_id']}, Source Name : {doc['metadata']['source_name']}"]
           
            combined_str = f"{', '.join(doc_strings)} , Score : {score}"
            formatted_items.append(combined_str)

            context_text = "\n\n".join(formatted_items)
 
    prompt = f"""You are a precise assistant that answers questions based ONLY on the provided context.
    
                `### Context
                {context_text}
    
                ### Instructions
                1. Answer the question using ONLY the information found in the Context above.
                2. If the Context does not contain the answer, explicitly say: "The provided context does not contain the answer." Do not guess, speculate, or use outside knowledge.
                3. For every claim you make, you MUST cite the specific account name and row ID from the source block. Format your citation exactly as: [Source: name, ID: id].
    
                ### Question
                {question}
    
                ### Answer:"""
    #print(get_Apikey())
    client = genai.Client(api_key =get_Apikey())
    response = client.models.generate_content(
          model ="gemini-3.5-flash",
          contents=prompt
       )
    print("output")
    print(response.text)

def bm25_search(question, k=5):
    # 1. Open and load the saved BM25 index and documents
    pickle_path = os.path.join(PERSIST_DIR, "bm25_index.pkl")

    with open(pickle_path, "rb") as f:
        payload = pickle.load(f)
        
    bm25_index = payload["index"]
    documents = payload["documents"]
    metadatas = payload["metaData"]
    
    # Tokenize and score the query
    tokenized_query = simple_tokenizer(question)
    scores = bm25_index.get_scores(tokenized_query)
    
    # Get top results
    k = min(k,len(documents))
    top_index = np.argsort(scores)[-k:][::-1]

    results = []
    for idx in top_index :
        results.append(
            {
                "index" : int(idx),
                "document": documents[idx],
                "score" : float(scores[idx]),
                "metadata": metadatas[idx]
            }
        )

    print("Top Results: BM25")
    for r in results:
        print(f"Index: {r['index']} | Score: {r['score']:.4f} | Document: {r['document'][:100]}... | Metadata: {r['metadata']}")


    return results
    

def query():
    question = "Redwood Networks 66"
    fromchroma = retreiveChromaDb(question)
    fromBM25 = bm25_search(question)
    fromreciprocal = reciprocal_rank_function([fromchroma, fromBM25])
    print(fromreciprocal)
    createPromptAndCallLLM(fromreciprocal,fromchroma ,fromBM25,question)


def reciprocal_rank_function(docList : list[list[str]], k=60):
    scores = {}
    for system_ranking in docList:
        for rank,doc_id in enumerate(system_ranking, start=1):
            if isinstance(doc_id, dict):
                metadata = doc_id.get("metadata")
            else:
                metadata = getattr(doc_id, "metadata", None)

            scores[metadata["supabase_id"]] = scores.get(metadata["supabase_id"],0) + 1/ (k+rank)
    return sorted(scores.items(), key=lambda x: x[1] , reverse=True)

query()