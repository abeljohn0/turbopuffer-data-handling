#!/usr/bin/env python3
"""
Modal-powered upload to Turbopuffer with parallel chunking and embedding generation.
Uses Modal to distribute chunking, embedding generation, and upload across multiple containers.
Optimized for performance with large documents.
"""

import json
import os
import time
import re
from typing import List, Dict, Any, Tuple
import modal
import random
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import turbopuffer
from tenacity import (
    retry,
    before_sleep_log,
    retry_if_exception_type,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
import logging
from tqdm import tqdm
from datetime import datetime


# Silence HuggingFace warnings and optimize tokenizer performance
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = modal.App("turbopuffer-upload")

class TransientUploadError(Exception):
    """Raised to trigger a retry for transient upload failures."""

# Modal image with all required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "transformers",
    "tiktoken", 
    "tqdm",
    "torch",
    "voyageai",
    "requests",
    "numpy",
    "turbopuffer",
    "tenacity"
])

# Shared storage for passing data between functions
storage = modal.Volume.from_name("turbopuffer-data", create_if_missing=True)

# Global tokenizer and client (reused across warm containers)
ENCODING = None
VOYAGE_CLIENT = None
CONTAINER_ID = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_globals(voyage_api_key: str):
    """Initialize global tokenizer and client once per container."""
    global ENCODING, VOYAGE_CLIENT, CONTAINER_ID

    if ENCODING is None:
        ENCODING = tiktoken.get_encoding("cl100k_base")
        print("✅ Global tokenizer initialized")
    # if TOKENIZER is None:
    #     from transformers import AutoTokenizer
    #     print("🔄 Loading Voyage Context 3 tokenizer globally...")
    #     try:
    #         TOKENIZER = AutoTokenizer.from_pretrained('voyageai/voyage-context-3')
    #         print("✅ Global tokenizer loaded")
    #     except Exception as e:
    #         print(f"⚠️ Voyage tokenizer failed: {e}, falling back to tiktoken")
    #         import tiktoken
    #         TOKENIZER = tiktoken.get_encoding("cl100k_base")
    
    if VOYAGE_CLIENT is None:
        import voyageai
        VOYAGE_CLIENT = voyageai.Client(api_key=voyage_api_key)
        print("✅ Global Voyage client initialized")

    CONTAINER_ID = os.environ.get("MODAL_TASK_ID", "unknown")

def count_tokens(text: str) -> int:
    """Fast token counting with global tokenizer."""
    # print("tiktoken count:", len(ENCODING.encode(text)))
    # print("voyage count:", VOYAGE_CLIENT.count_tokens([text], model="voyage-context-3"))
    # return len(ENCODING.encode(text))
    return VOYAGE_CLIENT.count_tokens([text], model="voyage-context-3")


# def semantic_segments(markdown: str) -> List[str]:
#     """Split markdown into semantic segments by headings and paragraphs."""
#     # Split on headings and blank lines
#     parts = re.split(r'(?m)^\s*#{1,6}\s+|(?:\n\s*\n)+', markdown)
#     return [p.strip() for p in parts if p.strip()]

def pack_segments(text: str, max_tokens: int = 800) -> Tuple[List[str], List[int]]:
    """Pack segments into chunks with token counts, avoiding massive tokenization."""
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=max_tokens,          # token size (adjust as needed)
    chunk_overlap=0,        # they recommend no overlap
    length_function=count_tokens,
    separators=["\n## ", "\n### ",
                "\n\n", 
                "\n- ", "\n* ", "\n• ",
                "\n", " ", ""],  # tries to split on paragraphs, then sentences, then words
    )
    chunks = text_splitter.split_text(text)
    chunk_token_counts = [count_tokens(chunk) for chunk in chunks]
    return chunks, chunk_token_counts

def _is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("429" in s) or ("rate" in s) or ("timeout" in s) or ("temporar" in s) or ("503" in s) or ("502" in s) or ("504" in s) or ("500" in s)

@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential_jitter(initial=30, max=180, exp_base=2, jitter=90),
    before_sleep=before_sleep_log(logger, logging.INFO),
    stop=stop_after_attempt(10),
    reraise=True,  # surface the real exception on final failure
)
def call_with_retry(cur_chunks: list[str]) -> list[list[float]]:
    # breakpoint()
    # print("voyage count:", VOYAGE_CLIENT.count_tokens(cur_chunks, model="voyage-context-3"))
    obj = VOYAGE_CLIENT.contextualized_embed(
        inputs=[cur_chunks],
        model="voyage-context-3",
        input_type="document",
    )
    return [e for r in obj.results for e in r.embeddings]

def embed_doc_chunks(chunks: List[str], chunk_token_counts: List[int]) -> List[List[float]]:
    """Embed document chunks with smart batching to stay under context limits."""
    MAX_DOC_CONTEXT = 30000  # max is 32k
    all_embeddings = []
    start = 0
    
    while start < len(chunks):
        cur, cur_tok = [], 0
        batch_start = start
        
        # Pack as many chunks as possible into current batch
        while start < len(chunks) and (cur_tok + chunk_token_counts[start]) <= MAX_DOC_CONTEXT:
            cur.append(chunks[start])
            cur_tok += chunk_token_counts[start]
            start += 1
        
        if not cur:
            # Single chunk is too large - skip it
            print(f"⚠️ Chunk at index {start} too large ({chunk_token_counts[start]} tokens), truncating")
            cur.append(chunks[start][:MAX_DOC_CONTEXT])
            cur_tok = MAX_DOC_CONTEXT
            start += 1
            # continue
        
        # Process this batch
        # if len(chunks) > 10:  # Only log for multi-chunk docs
        #     print(f"  Processing batch: chunks {batch_start}-{start-1} ({cur_tok} tokens)")
        
        batch_embeddings = call_with_retry(cur)
        time.sleep(0.1)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=7200,
    max_containers=200,  # Limit concurrent containers to avoid rate limits
)
def process_document_batch_with_embeddings(
    documents_batch: List[Dict[str, Any]], 
    voyage_api_key: str,
    turbopuffer_api_key: str,
    namespace: str,
    chunk_size: int = 800
) -> List[Dict[str, Any]]:
    """
    Process a batch of documents: chunk them and generate embeddings.
    Returns ready-to-upload chunk documents.
    """
    # Initialize globals once per container
    init_globals(voyage_api_key)
    
    def prepare_document_for_embedding(doc: Dict[str, Any]) -> Tuple[str, List[str], List[int]]:
        """Prepare document with semantic chunking and token counting."""
        text_parts = []
        
        if doc.get("displayName"):
            text_parts.append(f"Title: {doc['displayName']}")
        
        if doc.get("markdown"):
            text_parts.append(f"Content: {doc['markdown']}")
        
        combined_text = "\n\n".join(text_parts)
        
        # Smart semantic chunking
        # segments = semantic_segments(combined_text)
        chunks, chunk_token_counts = pack_segments(combined_text, chunk_size)
        
        return chunks, chunk_token_counts
    
    # Process this batch of documents
    all_chunk_docs = []
    batch_info = {
        'processed_docs': 0,
        'total_chunks': 0,
        'embedding_errors': 0,
        'mega_docs': 0
    }
    
    print(f"📄 Processing {len(documents_batch)} documents in this container...")
    done = 0
    total = len(documents_batch)
    # breakpoint()
    for doc in documents_batch:
        try:
            # Prepare and chunk the document
            chunks, chunk_token_counts = prepare_document_for_embedding(doc)
            # print("document length: ", len(combined_text), "total tokens: ", sum(chunk_token_counts))
            if not chunks:
                continue
            
            total_tokens = sum(chunk_token_counts)
            if total_tokens > 50000:  # Track mega-docs
                batch_info['mega_docs'] += 1
                print(f"🐳 Mega-doc detected: {total_tokens} tokens, {len(chunks)} chunks")
            
            # Generate embeddings for all chunks
            chunk_embeddings = embed_doc_chunks(chunks, chunk_token_counts)
            
            # Create Turbopuffer entries for each chunk
            for chunk_idx, (chunk_text, chunk_embedding, chunk_tokens) in enumerate(zip(chunks, chunk_embeddings, chunk_token_counts)):
                chunk_doc = {
                    "id": f"{doc.get('id')}_chunk_{chunk_idx}",
                    "original_doc_id": doc.get("id"),
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk_text,
                    "displayName": doc.get("displayName"),
                    "markdown": doc.get("markdown"),
                    "notesMarkdown": doc.get("notesMarkdown"),
                    "path": doc.get("path"),
                    "source": doc.get("source"),
                    "createdAt": doc.get("createdAt"),
                    "ind": doc.get("ind"),
                    "updatedAt": doc.get("updatedAt"),
                    "url": doc.get("url"),
                    # Chunk metadata (reusing calculated counts)
                    "total_chunks": len(chunks),
                    "total_doc_tokens": total_tokens,
                    "chunk_tokens": chunk_tokens,
                }
                
                # Add vector if embedding was generated
                if chunk_embedding is not None:
                    chunk_doc["vector"] = chunk_embedding
                else:
                    print(f"⚠️ Chunk at index {chunk_idx} has no embedding")
                    batch_info['embedding_errors'] += 1
                
                all_chunk_docs.append(chunk_doc)
            
            batch_info['processed_docs'] += 1
            batch_info['total_chunks'] += len(chunks)
            done += 1
            if (done % 100 == 0):
                pct = ((done / total) * 100) if total else 0.0
                print(f"Processed {done}/{total} docs - {pct:.1f}% complete in container {CONTAINER_ID}")
        except Exception as e:
            print(f"Error processing document {doc.get('id', 'unknown')}: {e}")
            continue
    
    print(f"✅ Container processed {batch_info['processed_docs']} docs → {batch_info['total_chunks']} chunks")
    if batch_info['mega_docs'] > 0:
        print(f"🐳 Handled {batch_info['mega_docs']} mega-docs")
    if batch_info['embedding_errors'] > 0:
        print(f"⚠️ {batch_info['embedding_errors']} embedding errors")
    
    print("Uploading to Turbopuffer...")
    result = chunk_and_upload(all_chunk_docs, batch_info['total_chunks'], 5000, turbopuffer_api_key, namespace)
    if result is not None and result["error"]:
        print(f"❌ Upload failed: {result['error']}")
    elif result is not None and result['rows_upserted']:
        print(f"✅ Uploaded {result['rows_upserted']} chunks successfully")
    return all_chunk_docs

# @app.function(
#     image=image,
#     cpu=1,
#     memory=2048,
#     timeout=600,
#     # Reduce upload concurrency to lessen 429/backlog pressure
#     max_containers=10,
# )

@retry(  # <-- Tenacity handles backoff + jitter + logging between attempts
    retry=retry_if_exception_type(TransientUploadError),
    wait=wait_exponential_jitter(initial=60, max=180, exp_base=2.5, jitter=100),
    before_sleep=before_sleep_log(logger, logging.INFO),
    stop=stop_after_attempt(12),
    reraise=True,  # surface the true final error if we give up
)
def _write_with_retry(ns, payload: Dict[str, Any]) -> int:
    """
    One write attempt to Turbopuffer.
    Raises TransientUploadError for retryable conditions so Tenacity will back off.
    Returns rows_upserted on success.
    """
    resp = ns.write(**payload)

    # Client returns a typed response with rows_upserted (per SDK docs)
    # https://github.com/turbopuffer/turbopuffer-python
    if resp is None:
        raise TransientUploadError("ns.write returned None")

    rows = getattr(resp, "rows_upserted", None)
    if isinstance(rows, int) and rows > 0:
        return rows

    status = getattr(resp, "status_code", None)
    text = getattr(resp, "text", "") or ""

    if status in (429, 500, 502, 503, 504):
        raise TransientUploadError(f"Retryable HTTP {status}: {text[:200]}")

    raise RuntimeError(f"Non-retryable write failure (status={status}): {text[:200]}")


def upload_chunks_to_turbopuffer(
    chunk_docs: List[Dict[str, Any]], 
    turbopuffer_api_key: str,
    namespace: str
) -> Dict[str, Any]:
    """Upload a batch of chunk documents to Turbopuffer."""
    import requests
    
    if not chunk_docs:
        return {"rows_upserted": 0, "error": "No chunks to upload"}
    
    base_url = f"https://gcp-us-west1.turbopuffer.com"
    headers = {
        "Authorization": f"Bearer {turbopuffer_api_key}",
        "Content-Type": "application/json"
    }

    client = turbopuffer.Client(api_key=turbopuffer_api_key, base_url=base_url)

    # try:
    ns = client.namespace(namespace=namespace)
    total_upserts = 0
    
    for chunks in chunk_docs:
        # Small pre-flight jitter to de-synchronize concurrent writers
        time.sleep(random.uniform(0.0, 2.0))

        # Prepare payload
        payload = {
            "upsert_rows": chunks,
            "distance_metric": "cosine_distance",
            "schema": {
            # Text content fields
            "chunk_text": {"type": "string", "full_text_search": True},
            "display_name": {"type": "string", "full_text_search": True},
            
            # ID and reference fields
            "original_doc_id": {"type": "string", "filterable": True},
            "chunk_index": {"type": "int", "filterable": False},
            "ind": {"type": "int", "filterable": True},
            
            # Metadata fields
            "path": {"type": "string", "filterable": True},
            "source": {"type": "string", "filterable": True},
            "url": {"type": "string", "filterable": True},
            "created_at": {"type": "uint", "filterable": True},
            "updated_at": {"type": "uint", "filterable": True},
            
            # Chunk statistics
            "total_chunks": {"type": "int", "filterable": False},
            "total_doc_tokens": {"type": "int", "filterable": True},
            "chunk_tokens": {"type": "int", "filterable": False},
        }
        }
        
        url = f"{base_url}/v2/namespaces/{namespace}"
        
        # Upload with robust retry logic
        try:
            rows = _write_with_retry(ns, payload)  # <-- Tenacity handles backoff/logging
            total_upserts += rows
        except RuntimeError as e:
            # Non-retryable error surfaced (schema issue, validation, etc.)
            return {"rows_upserted": total_upserts, "error": f"Non-retryable: {e}"}
        except Exception as e:
            # Tenacity exhausted retries on a transient error → surface it
            return {"rows_upserted": total_upserts, "error": f"Retryable error exhausted: {e}"}
    
    return {"rows_upserted": total_upserts, "error": None}

def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of documents."""
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return documents

def chunk_documents_for_processing(documents: List[Dict[str, Any]], batch_size: int = 50) -> List[List[Dict[str, Any]]]:
    """Split documents into batches for parallel processing."""
    batches = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batches.append(batch)
    return batches

def load_saved_chunks(file_path: str) -> List[Dict[str, Any]]:
    """Load previously saved chunk results."""
    import pickle
    import json
    
    print(f"📂 Loading saved chunks from {file_path}...")
    
    try:
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                chunks = pickle.load(f)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                chunks = json.load(f)
        else:
            raise ValueError("File must be .pkl or .json")
        
        print(f"✅ Loaded {len(chunks):,} chunks from {file_path}")
        return chunks
    
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return []

def chunk_vectorize_upload(documents: List[Dict[str, Any]], voyage_api_key: str, turbopuffer_api_key: str, namespace: str) -> List[Dict[str, Any]]:
    """Chunk and vectorize documents."""
    # Configuration
    chunk_size = 800  # tokens per chunk
    doc_batch_size = 1000  # documents per Modal container
    
    print(f"🎯 Configuration:")
    print(f"   Namespace: {namespace}")
    print(f"   Chunk size: {chunk_size} tokens")
    print(f"   Documents per container: {doc_batch_size}")
    # print(f"   Chunks per upload batch: {chunk_upload_batch_size}")
    
    # Split into batches for parallel processing
    doc_batches = chunk_documents_for_processing(documents, doc_batch_size)
    print(f"🚀 Processing {len(doc_batches)} batches in parallel on Modal...")
    
    # Process all document batches in parallel (chunking + embedding generation)
    print(f"\n🔄 Phase 1: Parallel chunking and embedding generation...")
    all_chunk_results = []
    
    for result in process_document_batch_with_embeddings.map(
        [batch for batch in doc_batches],
        [voyage_api_key] * len(doc_batches),
        [turbopuffer_api_key] * len(doc_batches),
        [namespace] * len(doc_batches),
        [chunk_size] * len(doc_batches)
    ):
        all_chunk_results.extend(result)
    
    total_chunks = len(all_chunk_results)
    print(f"✅ Generated {total_chunks:,} chunks with embeddings")
    
    # Save all_chunk_results to disk for safety
    import pickle
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Option 1: Save as pickle (preserves numpy arrays, faster)
    pickle_path = f"all_chunks.pkl"
    print(f"💾 Saving chunks to {pickle_path} for safety...")
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_chunk_results, f)
        print(f"✅ Chunks saved to {pickle_path}")
    except Exception as e:
        print(f"⚠️ Failed to save pickle: {e}")
    
    # # Option 2: Save as compressed JSON (human readable, larger)
    # json_path = f"all_chunks_{timestamp}.json"
    # print(f"💾 Saving chunks to {json_path} as backup...")
    # try:
    #     # Convert numpy arrays to lists for JSON serialization
    #     json_serializable = []
    #     for chunk in all_chunk_results:
    #         chunk_copy = chunk.copy()
    #         if 'vector' in chunk_copy and chunk_copy['vector'] is not None:
    #             chunk_copy['vector'] = chunk_copy['vector'].tolist() if hasattr(chunk_copy['vector'], 'tolist') else list(chunk_copy['vector'])
    #         json_serializable.append(chunk_copy)
        
    #     with open(json_path, 'w') as f:
    #         json.dump(json_serializable, f, indent=2)
    #     print(f"✅ JSON backup saved to {json_path}")
    # except Exception as e:
    #     print(f"⚠️ Failed to save JSON: {e}")
    
    # # Memory usage check
    # import psutil
    # process = psutil.Process(os.getpid())
    # memory_gb = process.memory_info().rss / (1024**3)
    # print(f"📊 Current memory usage: {memory_gb:.2f} GB")
    return all_chunk_results, total_chunks

def chunk_and_upload(all_chunk_results: List[Dict[str, Any]], total_chunks: int, chunk_upload_batch_size: int, turbopuffer_api_key: str, namespace: str):
    """Chunk and upload documents to Turbopuffer."""
        # Split chunks into upload batches
    chunk_upload_batches = []
    for i in range(0, total_chunks, chunk_upload_batch_size):
        batch = all_chunk_results[i:i + chunk_upload_batch_size]
        chunk_upload_batches.append(batch)
    
    print(f"\n🔄 Phase 2: Parallel upload to Turbopuffer...")
    print(f"   Uploading in {len(chunk_upload_batches)} batches...")
    
    # Upload all chunk batches in parallel
    total_uploaded = 0
    upload_errors = 0
    
    result = upload_chunks_to_turbopuffer(
        chunk_upload_batches,
        turbopuffer_api_key,
        namespace
    )
    if result is not None and result["error"]:
        print(f"❌ Upload failed: {result['error']}")
    elif result is not None and result['rows_upserted']:
        print(f"✅ Uploaded {result['rows_upserted']} chunks successfully")
        # uploaded_count = result.get('rows_upserted', 0)
        # total_uploaded += uploaded_count
        
        # if result.get('error'):
        #     upload_errors += 1
    
    # Final results
    print(f"\n🎉 Upload Complete!")
    # print(f"   📊 Total chunks uploaded: {total_uploaded:,}")
    # print(f"  num errors: {upload_errors}")

def find_remaining_docs(documents: List[Dict[str, Any]], turbopuffer_api_key: str, namespace: str) -> List[Dict[str, Any]]:
    """Find remaining documents that haven't been chunked and uploaded."""
    base_url = f"https://gcp-us-west1.turbopuffer.com"
    tpuf = turbopuffer.Turbopuffer(
        api_key=turbopuffer_api_key,
        region="gcp-us-west1"  # or whatever region your namespace is in
    )
    ns = tpuf.namespace(namespace=namespace)
    all_documents = []
    # for doc in ns.query(order_by={"attribute": "id", "direction": "asc"}).auto_pager():
    #     all_documents.append(doc)
    last_id = None
    while True:
        result = ns.query(
            rank_by=('id', 'asc'),
            top_k=1000,
            filters=('id', 'Gt', last_id) if last_id is not None else turbopuffer.NOT_GIVEN,
            include_attributes=["original_doc_id"],
        )
        # Do something with the page of results.
        all_documents.extend([row.original_doc_id for row in result.rows])
        if len(result.rows) < 1000:
            break
        last_id = result.rows[-1].id

    all_documents = set(all_documents)
    docs = [doc for doc in documents if doc["id"] not in all_documents]
    return docs

def get_all_docs(turbopuffer_api_key: str, namespace: str) -> List[Dict[str, Any]]:
    """Get all documents from Turbopuffer."""
    base_url = f"https://gcp-us-west1.turbopuffer.com"
    tpuf = turbopuffer.Turbopuffer(
        api_key=turbopuffer_api_key,
        region="gcp-us-west1"  # or whatever region your namespace is in
    )
    ns = tpuf.namespace(namespace=namespace)
    # ns2 = tpuf.namespace(namespace='content-prod')
    # ns2.delete_all()
    last_id = None
    all_documents = []
    while True:
        result = ns.query(
            rank_by=('id', 'asc'),
            top_k=1000,
            filters=('id', 'Gt', last_id) if last_id is not None else turbopuffer.NOT_GIVEN,
            include_attributes=True,
        )
        all_documents.extend(result.rows)
        if len(all_documents) >= 20000 or len(result.rows) < 1000:
            refactor_and_reupload(all_documents, tpuf)
            all_documents = []
        last_id = result.rows[-1].id

    # return all_documents

def refactor_and_reupload(all_docs: List[Dict[str, Any]], tpuf: turbopuffer.Turbopuffer):
    """Refactor and reupload documents to Turbopuffer."""
    base_url = f"https://gcp-us-west1.turbopuffer.com"
    print("entered refactor_and_reupload, chunks: ", len(all_docs))
    ns2 = tpuf.namespace(namespace='content-prod')
    chunks = []
    for chunk in tqdm(all_docs):
        chunk = chunk.model_dump()
        if "markdown" in chunk:
            del chunk["markdown"]

        if "notesMarkdown" in chunk:
            del chunk["notesMarkdown"]

        if "total_doc_tokens" in chunk:
            del chunk["total_doc_tokens"]
        chunk["updated_at"] = chunk.pop("updatedAt", chunk.get("updatedAt"))
        chunk["created_at"] = chunk.pop("createdAt", chunk.get("createdAt"))
        chunk["display_name"] = chunk.pop("displayName", chunk.get("displayName"))
        chunks.append(chunk)
        if len(chunks) >= 20000:
            payload = {
                "upsert_rows": chunks,
                "distance_metric": "cosine_distance",
                "schema": {
                    # Text content fields
                    "chunk_text": {"type": "string", "full_text_search": True},
                    "display_name": {"type": "string", "full_text_search": True},
                    
                    # ID and reference fields
                    "original_doc_id": {"type": "string", "filterable": True},
                    "chunk_index": {"type": "int", "filterable": False},
                    "ind": {"type": "int", "filterable": True},
                    
                    # Metadata fields
                    "path": {"type": "string", "filterable": True},
                    "source": {"type": "string", "filterable": True},
                    "url": {"type": "string", "filterable": True},
                    "created_at": {"type": "uint", "filterable": True},
                    "updated_at": {"type": "uint", "filterable": True},
                    
                    # Chunk statistics
                    "total_chunks": {"type": "int", "filterable": False},
                    "chunk_tokens": {"type": "int", "filterable": False},
                }
            }
            _write_with_retry(ns2, payload)
            print(f"✅ Uploaded {len(chunks):,} chunks successfully")
            chunks = []
            time.sleep(2)
    if (len(chunks) > 0):
        payload = {
            "upsert_rows": chunks,
            "distance_metric": "cosine_distance",
            "schema": {
                # Text content fields
                "chunk_text": {"type": "string", "full_text_search": True},
                "display_name": {"type": "string", "full_text_search": True},
                
                # ID and reference fields
                "original_doc_id": {"type": "string", "filterable": True},
                "chunk_index": {"type": "int", "filterable": False},
                "ind": {"type": "int", "filterable": True},
                
                # Metadata fields
                "path": {"type": "string", "filterable": True},
                "source": {"type": "string", "filterable": True},
                "url": {"type": "string", "filterable": True},
                "created_at": {"type": "uint", "filterable": True},
                "updated_at": {"type": "uint", "filterable": True},
                
                # Chunk statistics
                "total_chunks": {"type": "int", "filterable": False},
                "chunk_tokens": {"type": "int", "filterable": False},
            }
        }
        _write_with_retry(ns2, payload)
        print(f"✅ Uploaded {len(chunks):,} chunks successfully")
        # ns.update(doc["id"], doc)

def upload_companies_to_turbopuffer(documents: List[Dict[str, Any]], turbopuffer_api_key: str):
    """Upload companies to Turbopuffer."""
    tpuf = turbopuffer.Turbopuffer(
        api_key=turbopuffer_api_key,
        region="gcp-us-west1"
    )
    ns2 = tpuf.namespace(namespace='public-companies')
    # ns2.delete_all()
    docs = []
    for doc in tqdm(documents):
        if 'text_match_info' in doc:
            del doc['text_match_info']
        if 'displayName' in doc:
            doc['display_name'] = doc['displayName']
            del doc['displayName']
        if 'tickers' in doc:
            tickers = doc['tickers']
            new_tickers = []
            for ticker in tickers:
                new_tickers.append(f"{ticker["exchange"]}_{ticker["ticker"]}")
            doc['tickers'] = new_tickers
        if 'createdAt' in doc:
            dt = datetime.fromisoformat(doc['createdAt'].replace("Z", "+00:00"))
            createdAt_ms = int(dt.timestamp() * 1000)
            doc['created_at'] = createdAt_ms
            del doc['createdAt']
        if 'updatedAt' in doc:
            doc['updated_at'] = doc['updatedAt']
            del doc['updatedAt']
        if 'backlinkUrl' in doc:
            doc['backlink_url'] = doc['backlinkUrl']
            del doc['backlinkUrl']
            
        docs.append(doc)
    payload = {
        "upsert_rows": docs,
        "schema": {
            # Text content fields
            "name": {"type": "string", "full_text_search": True},
            "display_name": {"type": "string", "full_text_search": True},
            "isins": {"type": "[]string", "filterable": True},
            
            # ID and reference fields
            "country": {"type": "string", "filterable": True},
            "tickers": {"type": "[]string", "filterable": True},
            "ind": {"type": "int", "filterable": True},
            
            # Metadata fields
            "backlink_url": {"type": "string", "filterable": False},
            "created_at": {"type": "uint", "filterable": True},
            "updated_at": {"type": "uint", "filterable": True},
        }
    }
    print(docs[:5])
    ns2.write(**payload)
    print(f"✅ Uploaded {len(documents):,} documents successfully")

# @app.local_entrypoint()
def main():
    """
    Main orchestration function that runs locally and coordinates Modal functions.
    """
    use_saved_chunks = os.getenv("USE_SAVED_CHUNKS") == "1"
    
    # Check API keys
    turbopuffer_api_key = os.getenv("TURBOPUFFER_API_KEY")
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    
    if not turbopuffer_api_key:
        print("Error: Please set TURBOPUFFER_API_KEY environment variable")
        return
    
    if not voyage_api_key:
        print("Error: Please set VOYAGE_API_KEY environment variable")
        return
    
    chunk_upload_batch_size = 5000
    # namespace = "content-prod-backup"

    # File path
    jsonl_file_path = "/Users/abeljohn/Downloads/documents-public-companies.jsonl"
    
    # Read documents locally
    # print(f"📖 Reading documents from {jsonl_file_path}...")
    documents = read_jsonl_file(jsonl_file_path)
    upload_companies_to_turbopuffer(documents, turbopuffer_api_key)
    # if not documents:
    #     print("❌ No documents found")
    #     return
    
    # all_docs = get_all_docs(turbopuffer_api_key, namespace)
    # refactor_and_reupload(all_docs, turbopuffer_api_key)
    # print(f"✅ Found {len(all_docs):,} documents in Turbopuffer")

    # if not use_saved_chunks:
    # print(f"Retrieved {len(documents):,} documents in the jsonl file")
    # docs = find_remaining_docs(documents, turbopuffer_api_key, namespace)
    # print(f"✅ Found {len(docs):,} documents to chunk and upload")
    # all_chunk_results, total_chunks = chunk_vectorize_upload(docs, voyage_api_key, turbopuffer_api_key, namespace)
    # else:
    #     all_chunk_results = load_saved_chunks("/Users/abeljohn/Developer/accordance-projects/vectorDB-vs-contextWindow/all_chunks.pkl")
    #     total_chunks = len(all_chunk_results)
    #     chunk_and_upload(all_chunk_results, total_chunks, chunk_upload_batch_size, turbopuffer_api_key, namespace)
        # all_chunk_results = load_saved_chunks("/Users/abeljohn/Developer/accordance-projects/vectorDB-vs-contextWindow/all_chunks_most_recent.pkl")
    #     doc_ids = set([chunk["original_doc_id"] for chunk in all_chunk_results])
    #     documents_left = [doc for doc in documents if doc["id"] not in doc_ids]
    #     if len(documents_left) > 0:
    #         print(f"{len(documents_left):,} documents left to chunk and upload")
    #         documents_chunks_left, _ = chunk_and_vectorize(documents_left, voyage_api_key)
    #         all_chunk_results.extend(documents_chunks_left)
    #         import pickle
    #         pickle_path = f"all_chunks_most_recent.pkl"
    #         print(f"💾 Saving chunks to {pickle_path} for safety...")
    #         try:
    #             with open(pickle_path, 'wb') as f:
    #                 pickle.dump(all_chunk_results, f)
    #             print(f"✅ Chunks saved to {pickle_path}")
    #         except Exception as e:
    #             print(f"⚠️ Failed to save pickle: {e}")
    #         total_chunks = len(documents_chunks_left)
    #         chunk_and_upload(documents_chunks_left, total_chunks, chunk_upload_batch_size, turbopuffer_api_key, namespace)
    #         return
    #     else:
    #         print("✅ All documents have been chunked and uploaded")
    #         return
    # total_chunks = len(all_chunk_results)
    # chunk_and_upload(all_chunk_results, total_chunks, chunk_upload_batch_size, turbopuffer_api_key, namespace)

if __name__ == "__main__":
    main()
