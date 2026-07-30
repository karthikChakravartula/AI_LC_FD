import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai
import os
from dotenv import load_dotenv 

def get_Apikey():
    load_dotenv()#dotenv_path=current_dir / ".env")
    return os.getenv("GEMINI_API_KEY")

def retreiveChromaDb(question) :
   embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   db = Chroma(persist_directory="./chroma_db_storage", embedding_function=embedding_model)
   results = db.similarity_search(question, k=3)
   raw_text_chunks = [f"{doc.page_content} , Name : {doc.metadata.get("source_name")}, RowId : {doc.metadata.get("supabase_id")}" for doc in results]
   #print("\n\n".join(raw_text_chunks))
   combined_context = "\n\n".join(raw_text_chunks)
   #print(combined_context)
   prompt = f"""You are a precise assistant that answers questions based ONLY on the provided context.

            `### Context
            {combined_context}

            ### Instructions
            1. Answer the question using ONLY the information found in the Context above.
            2. If the Context does not contain the answer, explicitly say: "The provided context does not contain the answer." Do not guess, speculate, or use outside knowledge.
            3. For every claim you make, you MUST cite the specific account name and row ID from the source block. Format your citation exactly as: [Source: name, ID: id].

            ### Question
            {question}

            ### Answer:"""
   print(get_Apikey())
   client = genai.Client(api_key =get_Apikey())
   response = client.models.generate_content(
      model ="gemini-3.5-flash",
      contents=prompt
   )
   print("output")
   print(response.text)

#    formatted_context = ""
#    for doc in results:
#     # Extract metadata safely with fallbacks
#         name = doc.metadata.get("source_name", "Unknown Account")
#         row_id = doc.metadata.get("supabase_id", "Unknown ID")
#         text = doc.page_content
    
#     # Append the formatted chunk with clean delimiters
#         formatted_context += f"[Source: {name}, ID: {row_id}]\n{text}\n\n"

#retreiveChromaDb("tell me weather in chicago")