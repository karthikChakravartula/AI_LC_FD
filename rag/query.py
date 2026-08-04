import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai
import os
from dotenv import load_dotenv 
from lib.common import simple_tokenizer
import pickle
import numpy as np

PERSIST_DIR = "./chroma_db_storage"

def get_Apikey():
    load_dotenv()#dotenv_path=current_dir / ".env")
    return os.getenv("GEMINI_API_KEY")

def retreiveChromaDb(question , k = 5) :
   embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
   res = db.similarity_search(question, k)
   print(f"From Chroma \n\n {'\n\n'.join([r.page_content for r in res])}")
   return res

def createPromptAndCallLLM(ReciprocalRank :list[str], chromaList : list[str] , question):

    formatted_items = []

    for doc_id, score in ReciprocalRank[:5]:
            # 1. Find matching documents
            matches = [r for r in chromaList if r.metadata.get("supabase_id") == doc_id]
            
            # 2. Format the matching documents
            doc_strings = [
                f"Content : {r.page_content}, Id : {r.metadata.get('supabase_id')}, Source Name : {r.metadata.get('source_name')}"
                for r in matches
            ]
            
            # 3. Combine with the score
            combined_str = f"{', '.join(doc_strings)} , Score : {score}"
            formatted_items.append(combined_str)

            context_text = "\n\n".join(formatted_items)
       #context_text = "\n\n".join([f"{[f"Content : {r.page_content}, Id : {r.metadata.get("supabase_id")}, Source Name : {r.metadata.get("source_name")}" for r in chromaList if r.metadata.get("supabse_id") == doc_id]} , Score : {score}" for doc_id, score in ReciprocalRank[:5]])



       #raw_text_chunks = [f"{doc.page_content} , Name : {doc.metadata.get("source_name")}, RowId : {doc.metadata.get("supabase_id")}" for doc in results]
       #print("\n\n".join(raw_text_chunks))
       #combined_context = "\n\n".join(raw_text_chunks)
       #print(combined_context)
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

    # with open(pickle_path, "rb") as f:
    #     bm25_index, processed_documents = pickle.load(f)

    # print(f"Index corpus size: {bm25_index.corpus_size}")
    # print(f"Processed documents size: {len(processed_documents)}")

    # # 2. Clean and chop the user's question
    # tokenized_query = simple_tokenizer(question)

    # # 3. Ask BM25 to find and return the top-k matching documents
    # top_docs = bm25_index.get_top_n(
    #     tokenized_query, processed_documents, n=k
    # )

    # all_scores = bm25_index.get_scores(tokenized_query)

    # for doc in top_docs:
    #     # Find where this document lives in the original list to grab its score
    #     doc_index = processed_documents.index(doc)
    #     # Attach the score property dynamically
    #     score = all_scores[doc_index]

    #     print(f"Content \n {doc}\n{doc_index}\n{score}\n\n")


    # return top_docs

def query():
    question = "Any Nonprofit-Affiliated Organization specializing in media streaming?"
    fromchroma = retreiveChromaDb(question)
    fromBM25 = bm25_search(question)
    fromreciprocal = reciprocal_rank_function([fromchroma, fromBM25])
    print(fromreciprocal)
    createPromptAndCallLLM(fromreciprocal,fromchroma ,question)


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