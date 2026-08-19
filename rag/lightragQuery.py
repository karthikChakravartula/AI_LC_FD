from lightrag import LightRAG
import os
import asyncio
from lightrag.llm.gemini import gemini_model_complete, gemini_complete_if_cache
from dotenv import load_dotenv 
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc 
import google.genai as genai
import numpy as np
from google.genai import types
import shutil;

load_dotenv()
WORKING_DIR = "./local_neo4j_workdir"

shutil.rmtree(WORKING_DIR, ignore_errors=True)
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

def lightragLoading() :
    print(f"API_KEY: {os.getenv("NEO4J_URI")}")
    #llm_func = partial(gemini_complete_if_cache, model="gemini-3.5-flash")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        # Brain: Set up Gemini 3.5 Flash for text completion
        #llm_model_func=llm_func,
        llm_model_func=gemini_model_complete,
        
        #llm_model_kwargs={"model": "gemini-3.5-flash"},
        llm_model_name="gemini-3.5-flash", 
        
        # Translator: Set up text-embedding-004 for graph indexing
        embedding_func=EmbeddingFunc(
            embedding_dim=768,       # Dimension size for text-embedding-004
            max_token_size=8192,     # Token window for the model
            func=gemini_embed        # Your original custom gemini function
        ),
        
        #embedding_kwargs={"model": "text-embedding-004"},
        
        # Backend Storage
        graph_storage="Neo4JStorage",
        log_level="DEBUG"
    )
    return rag

async def query() :
    rag = lightragLoading()
    print("Initializing storage backends...")
    await rag.initialize_storages()
    
    print("Initializing pipeline processing status...")
    await initialize_pipeline_status()

    # ==========================================
    print("\n--- Step 1: Ingesting text into LightRAG ---")
    test_text = """
    Quantum computing is a rapidly emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers. 
    The field of quantum computing was originally inspired by Richard Feynman in 1982. 
    Today, companies like IBM and Google are building advanced quantum processors using superconducting qubits to achieve quantum supremacy.
    """

    # Ingest the text (this populates both Neo4j and your local vector JSON files)
    await rag.ainsert(test_text)
    print("Insertion completed successfully!")

    try:
        await rag.ainsert(test_text)
        print("Insertion completed successfully!")
    except Exception as e:
        import traceback
        print(f"INSERTION FAILED: {e}")
        traceback.print_exc()

    # ==========================================
    # 4. TEST STEP 2: QUERY THE PIPELINE
    # ==========================================
    print("\n--- Step 2: Querying the Knowledge Graph ---")
    query_text = "Who inspired the field of quantum computing and what technology is being used today?"

    # Run a hybrid search query (combines vector search and graph search)
    response = await rag.aquery(query_text)

    print(f"Query: {query_text}")
    print(f"Response:\n{response}")

_embed_call_count = 0
async def gemini_embed(texts: list[str]) -> list[list[float]]:
    # 1. Defensively handle if LightRAG passes a single raw string instead of a list
    global _embed_call_count
    _embed_call_count += 1
    print(f"[gemini_embed call #{_embed_call_count}] received {len(texts) if isinstance(texts, list) else 1} texts")
    if isinstance(texts, str):
        texts = [texts]
        
    client = genai.Client()
    
    # 2. Call the Google GenAI client
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    
    # 3. Safely extract vectors. 
    # text-embedding-004 returns a list of embedding objects matching the input size.
    results = []
    for emb in response.embeddings:
        # Extract the raw 768-dimensional float list
        results.append(list(emb.values))

    print(f"Embedded {len(results)} vectors, dim={len(results[0]) if results else 'empty'}")
    # 4. Strict structural check before returning to LightRAG's EmbeddingFunc
    if len(results) != len(texts):
        # Troubleshooting safeguard: if it still mismatches, truncate or pad
        # to guarantee the outer framework doesn't crash.
        print(f"[WARN] Expected {len(texts)} vectors but got {len(results)}. Fixing...")
        if len(results) > len(texts):
            results = results[:len(texts)]
            
    return np.array(results, dtype=np.float32)

asyncio.run(query())