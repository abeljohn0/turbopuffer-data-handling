# verify_turbopuffer_vectors.py
import os, random, pickle, glob, collections, time, re
import turbopuffer as tpuf
import voyageai
import anthropic
import json
import pickle
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
import typesense
import csv

import dotenv
dotenv.load_dotenv()

anthro_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# @retry(
#     stop=stop_after_attempt(5),                 # max 5 tries
#     wait=wait_random_exponential(multiplier=1, max=60),  # backoff
#     reraise=True
# )
def call_anthro_with_retry(client, prompt):
    """Call Anthropic API and ensure valid JSON is returned."""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    text = message.content[0].text.strip()
    # sanity check
    if not text:
        raise ValueError("Empty response from model")

    try:
        jsonMatch = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if not jsonMatch:
            jsonMatch = [None, text]
        json_text = jsonMatch[1].strip()
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        # force a retry by raising an exception
        print(f"Invalid JSON: {text}")
        raise ValueError(f"Invalid JSON: {text}...") from e

def build_enhancement_prompt(original_query: str) -> str:
    """Build the enhancement prompt for Claude matching the JS version."""
    return (
        "You are an expert at enhancing search queries for tax, accounting, and finance professionals using the Accordance AI system. "
        "Your task is to take a user's search query and optimize it for two different search engines:"
        "1. **Typesense** - A keyword-based text search engine that works best with expanded keywords and synonyms"
        "2. **Turbopuffer** - A semantic vector search engine that works best with natural language and contextual queries"
        "# User's Original Query:\n"
        f"\"{original_query}\"\n\n"
        "# Your Task:\n"
        "Generate a JSON response with optimized queries for both search engines, following this exact format:\n\n"
        "```json"
        "{"
        "  \"typesense_query\": \"Enhanced keyword-rich query for text search\","
        "  \"turbopuffer_query\": \"Enhanced natural language query for semantic search\","
        "  \"reasoning\": \"Brief explanation of enhancements made\""
        "}"
        "```"
        "# Enhancement Guidelines:"
        "## For Typesense (Keyword Search):"
        "- Add relevant synonyms and related terms"
        "- Include legal/tax terminology variations"
        "- Add document type keywords (section, rule, regulation, statute, code, etc.)"
        "- Expand abbreviations (e.g., \"IRS\" → \"IRS Internal Revenue Service\")"
        "- Include plural/singular variations"
        "- Add industry-specific terms"
        "## For Turbopuffer (Semantic Search):"
        "- Create natural, contextual language"
        "- Add legal/regulatory context when appropriate"
        "- Preserve the query's intent while making it more descriptive"
        "- Include relevant domain context (tax, accounting, legal, etc.)"
        "- Make the query more specific and actionable"
        "Query: \"tax deduction\""
        "- Typesense: \"tax deduction deductions write-off allowance exemption IRS Internal Revenue Service section code regulation\""
        "- Turbopuffer: \"tax deduction regulations and allowable write-offs under Internal Revenue Code\""
        "Query: \"AICPA ethics\""
        "- Typesense: \"AICPA ethics ethical standards professional conduct code CPAs accountants\""
        "- Turbopuffer: \"AICPA professional ethics standards and code of conduct for certified public accountants\""
        "Now enhance the user's query following these guidelines."
    )

def enhance_query(original_query: str) -> dict:
    """Enhance a user's query for Typesense and Turbopuffer using Claude.

    Returns a dict with keys: original, typesense, turbopuffer.
    """
    if not anthro_client:
        raise RuntimeError("Anthropic client not initialized; set ANTHROPIC_API_KEY")

    prompt = build_enhancement_prompt(original_query)

    try:
        message = anthro_client.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )

        content = message.content[0].text
        m = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        json_text = (m.group(1) if m else content).strip()
        enhancement = json.loads(json_text)

        return {
            "original": original_query,
            "typesense": {
                "keywordQuery": enhancement.get("typesense_query", "")
            },
            "turbopuffer": {
                "keywordQuery": enhancement.get("typesense_query", ""),
                "enhancedQuery": enhancement.get("turbopuffer_query", ""),
            },
        }
    except Exception as e:
        # Mirror JS behavior: log and return None/empty fallback
        print(f"LLM enhancement failed {e}")

def time_fn_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return res, t1 - t0

def ask_questions(samples, n_samples):
    for idx in tqdm(range(1, len(samples) + 1)):
        questions = enhance_query(samples[idx]["question"])
        typesense_results, t1 = time_fn_call(search_client.typesense_search, questions["typesense"]["keywordQuery"])
        vector_search_results, t2 = time_fn_call(search_client.vector_search, questions["turbopuffer"]["enhancedQuery"])
        hybrid_search_results, t3 = time_fn_call(search_client.hybrid_search, questions["turbopuffer"]["enhancedQuery"], questions["turbopuffer"]["keywordQuery"])
        full_text_search_results, t4 = time_fn_call(search_client.full_text_search, questions["turbopuffer"]["keywordQuery"])
        vector_search_reranked_results, t5 = time_fn_call(search_client.vector_search_with_reranker, questions["turbopuffer"]["enhancedQuery"])
        hybrid_search_reranked_results, t6 = time_fn_call(search_client.hybrid_search_with_reranker, questions["turbopuffer"]["enhancedQuery"], questions["turbopuffer"]["keywordQuery"])
        samples[idx]["typesense_results"] = {
            "results": typesense_results,
            "elapsed": t1
        }
        samples[idx]["vector_search_results"] = {
            "results": vector_search_results,
            "elapsed": t2
        }
        samples[idx]["hybrid_search_results"] = {
            "results": hybrid_search_results,
            "elapsed": t3
        }
        samples[idx]["full_text_search_results"] = {
            "results": full_text_search_results,
            "elapsed": t4
        }
        samples[idx]["vector_search_reranked_results"] = {
            "results": vector_search_reranked_results,
            "elapsed": t5
        }
        samples[idx]["hybrid_search_reranked_results"] = {
            "results": hybrid_search_reranked_results,
            "elapsed": t6
        }
        time.sleep(random.random() * 3)
    pickle_path = f"questions_and_answers_{n_samples}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(samples, f)
    print(f"💾 Saved {len(samples)} questions and answers to {pickle_path}")


def evaluate_results(q_and_a, n_samples):
    counts = {
        "typesense": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "vector": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "hybrid": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "full_text": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "vector_reranked": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "hybrid_reranked": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
    }
    chunk_counts = {
        "vector": 0,
        "hybrid":0,
        "full_text": 0,
        "vector_reranked": 0,
        "hybrid_reranked": 0,
    }
    avg_times = {
        "typesense": 0,
        "vector": 0,
        "hybrid": 0,
        "full_text": 0,
        "vector_reranked": 0,
        "hybrid_reranked": 0,
    }
    for k, test in q_and_a.items():
        correct_doc_id = test['doc_id']
        chunk_correct = []
        ts_correct = []
        vs_correct = []
        hs_correct = []
        fts_correct = []
        vsr_correct = []
        hsr_correct = []
        for ts, vs, hs, fts, vsr, hsr in zip(test['typesense_results']['results'], test['vector_search_results']['results'], test['hybrid_search_results']['results'], test['full_text_search_results']['results'], test['vector_search_reranked_results']['results'], test['hybrid_search_reranked_results']['results']):
            if correct_doc_id == ts['doc_id']:
                counts['typesense'][ts['rank']] += 1
                ts_correct.append(ts['rank'])
            if correct_doc_id == vs['doc_id']:
                counts['vector'][vs['rank']] += 1
                vs_correct.append(vs['rank'])
                if test['chunk_index'] == vs['chunk_index']:
                    chunk_counts['vector'] += 1
                    chunk_correct.append(('vs', vs['rank']))
            if correct_doc_id == hs['doc_id']:
                counts['hybrid'][hs['rank']] += 1
                hs_correct.append(hs['rank'])
                if test['chunk_index'] == hs['chunk_index']:
                    chunk_counts['hybrid'] += 1
                    chunk_correct.append(('hs', hs['rank']))
            if correct_doc_id == fts['doc_id']:
                counts['full_text'][fts['rank']] += 1
                fts_correct.append(fts['rank'])
                if test['chunk_index'] == fts['chunk_index']:
                    chunk_counts['full_text'] += 1
                    chunk_correct.append(('fts', fts['rank']))
            if correct_doc_id == vsr['doc_id']:
                counts['vector_reranked'][vsr['rank']] += 1
                vsr_correct.append(vsr['rank'])
                if test['chunk_index'] == vsr['chunk_index']:
                    chunk_counts['vector_reranked'] += 1
                    chunk_correct.append(('vsr', vsr['rank']))
            if correct_doc_id == hsr['doc_id']:
                counts['hybrid_reranked'][hsr['rank']] += 1
                hsr_correct.append(hsr['rank'])
                if test['chunk_index'] == hsr['chunk_index']:
                    chunk_counts['hybrid_reranked'] += 1
                    chunk_correct.append(('hsr', hsr['rank']))
        q_and_a[k]['chunk_correct'] = chunk_correct
        q_and_a[k]['correct'] = {
            'ts': ts_correct,
            'vs': vs_correct,
            'hs': hs_correct,
            'fts': fts_correct,
            'vsr': vsr_correct,
            'hsr': hsr_correct,
        }
        avg_times['typesense'] += test['typesense_results']['elapsed']
        avg_times['vector'] += test['vector_search_results']['elapsed']
        avg_times['hybrid'] += test['hybrid_search_results']['elapsed']
        avg_times['full_text'] += test['full_text_search_results']['elapsed']
        avg_times['vector_reranked'] += test['vector_search_reranked_results']['elapsed']
        avg_times['hybrid_reranked'] += test['hybrid_search_reranked_results']['elapsed']
    print('overall document hits:', [(k, sum(v.values())) for k, v in counts.items()])
    print('document hits by rank:', counts)

    for k, v in avg_times.items():
        avg_times[k] /= len(q_and_a)
    print('average times:', avg_times)
    print('chunk hits:', chunk_counts)

    pickle_path = f"q_and_a_eval_{n_samples}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(q_and_a, f)

def generate_questions_from_chunk(n_samples):
    prompt ="""
    You are a tax expert that is testing a document search tool by asking questions related to document excerpts.
    Given the following 10 document excerpts, generate 10 questions that are relevant to the document while still being general enough to be asked to the search tool.
    Here is an example:
    Sample Text:
    Title: Section 305: Limited duties of limited partners

    Sample Content: 
    1.  {{< pincite identifier="a" display="a" >}} A limited partner does not have any fiduciary duty to the limited partnership or to any other partner solely by reason of being a limited partner.
    1.  {{< pincite identifier="b" display="b" >}} A limited partner shall discharge the duties to the partnership and the other partners under this Act or under the partnership agreement and exercise any rights consistently with the obligation of good faith and fair dealing.
    1.  {{< pincite identifier="c" display="c" >}} A limited partner does not violate a duty or obligation under this Act or under the partnership agreement merely because the limited partner's conduct furthers the limited partner's own interest.
    { .parens-lower-alpha }

    Sample Question: 
    What fiduciary duties and obligations do limited partners have under Section 305? 

    # Your Task:
    Generate a JSON response with the following fields:
    [
        {
            "q_no": "Question number provided in the input",
            "question": "Question text",
        },
        {
            "q_no": "Question number provided in the input",
            "question": "Question text",
        },
        ...
    ]

    Here are your formatted document excerpts:
    """
    samples = search_client.ns.query(
        include_attributes=True,
        top_k=1000
    )
    samples = random.sample(samples.rows, n_samples)
    question_samples = collections.defaultdict(dict)
    for idx in tqdm(range(0, n_samples, 10)):
        input_q = []
        for i, sample in enumerate(samples[idx:idx+10]):
            q_no = idx + i + 1
            question_samples[q_no] = {
                "doc_id": sample["original_doc_id"],
                "chunk_text": sample["chunk_text"],
                "chunk_index": sample["chunk_index"],
            }
            input_q.append({"q_no": q_no, "chunk_text": sample["chunk_text"]})

        # build prompt once per block
        block_prompt = prompt + json.dumps(input_q) + "\n\nRemember, only return JSON."
        try:
            responses = call_anthro_with_retry(anthro_client, block_prompt)
        except Exception as e:
            print(f"⚠️ Failed after retries at idx={idx}: {e}")
            breakpoint()
            # continue   # skip this block instead of crashing
        print([x["q_no"] for x in responses])
        for resp in responses:
            question_samples[int(resp["q_no"])]["question"] = resp["question"]
        time.sleep(random.randint(1, 3))
    print(f"💾 Saving {len(question_samples)} questions to sample_questions.pkl")
    pickle_path = f"sample_questions_{n_samples}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(question_samples, f)
    # breakpoint()

class SearchFunctions:
    def __init__(self, name_space="content-prod"):
        self.voy = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.ts_client = typesense.Client({
            'nodes': [{
                'host': os.getenv("TYPESENSE_HOST"),  # e.g., 'localhost' or your Typesense Cloud hostname
                'port': os.getenv("TYPESENSE_PORT"),                 # or '443' for HTTPS
                'protocol': os.getenv("TYPESENSE_PROTOCOL")              # or 'https'
            }],
            'api_key': os.getenv("TYPESENSE_API_KEY"),          # Your Typesense API key
            'connection_timeout_seconds': 30
        })
        self.tp = tpuf.Client(api_key=os.getenv("TURBOPUFFER_API_KEY"), region="gcp-us-west1")
        self.name_space = name_space
        self.ns = self.tp.namespace(self.name_space)

    def vector_search(self, enhanced_query):
        qe = self.voy.contextualized_embed(inputs=[[enhanced_query]], model="voyage-context-3", input_type="query").results[0].embeddings[0]
        res_vec = self.ns.query(
            rank_by=("vector", "ANN", qe),
            top_k=5,
            include_attributes=True,
        )
        results = []
        for i, row in enumerate(res_vec.rows, 1):
            row_dict = row.model_dump()
            results.append({
                "rank": i,
                "confidence": 1 - row_dict['$dist'],
                "doc_id": row_dict['original_doc_id'],
                "chunk_index": row_dict['chunk_index'],
                "chunk_text": row_dict['chunk_text'],
            })
        return results

    def vector_search_with_reranker(self, enhanced_query):
        qe = self.voy.contextualized_embed(inputs=[[enhanced_query]], model="voyage-context-3", input_type="query").results[0].embeddings[0]
        res_vec = self.ns.query(
            rank_by=("vector", "ANN", qe),
            top_k=50,
            include_attributes=True,
        )

        res_vec_text = []

        for i, row in enumerate(res_vec.rows, 1):
            res_vec_text.append(row.chunk_text)

        reranking = self.voy.rerank(enhanced_query, res_vec_text, model="rerank-2.5", top_k=5)
        results = []
        for i, res in enumerate(reranking.results, 1):
            og_index = res[0] 
            results.append({
                "rank": i,
                "confidence": 1 - res_vec.rows[og_index]['$dist'],
                "doc_id": res_vec.rows[og_index]['original_doc_id'],
                "chunk_index": res_vec.rows[og_index]['chunk_index'],
                "chunk_text": res_vec.rows[og_index]['chunk_text'],
            })
        return results


    def hybrid_search(self, enhanced_query, keyword_query):
        qe = self.voy.contextualized_embed(inputs=[[enhanced_query]], model="voyage-context-3", input_type="query").results[0].embeddings[0]
        res_vec = self.ns.multi_query(
            queries=[
            {
                "rank_by": ("vector", "ANN", qe),
                "top_k": 5,
                "include_attributes": True,
            },
                {
                "rank_by": ("chunk_text", "BM25", keyword_query),
                "top_k": 5,
                "include_attributes": True,
                }
            ]
        )
        fused_results = self.reciprocal_rank_fusion([res_vec.results[0].rows, res_vec.results[1].rows])
        results = []
        for i, obj in enumerate(fused_results[:5], 1):
            row_dict = obj.model_dump()
            results.append({
            "rank": i,
            "confidence": 1 - row_dict['$dist'],
                "doc_id": row_dict['original_doc_id'],
                "chunk_index": row_dict['chunk_index'],
                "chunk_text": row_dict['chunk_text'],
            })
        return results

    def hybrid_search_with_reranker(self, enhanced_query, keyword_query):
        qe = self.voy.contextualized_embed(inputs=[[enhanced_query]], model="voyage-context-3", input_type="query").results[0].embeddings[0]
        res_vec = self.ns.multi_query(
            queries=[
            {
                "rank_by": ("vector", "ANN", qe),
                "top_k": 50,
                "include_attributes": True,
            },
                {
                "rank_by": ("chunk_text", "BM25", keyword_query),
                "top_k": 50,
                "include_attributes": True,
            }
            ]
        )
        fused_results = self.reciprocal_rank_fusion([res_vec.results[0].rows, res_vec.results[1].rows])

        res_vec_text = []
        for i, row in enumerate(fused_results, 1):
            res_vec_text.append(row.chunk_text)

        reranking = self.voy.rerank(enhanced_query, res_vec_text, model="rerank-2.5", top_k=5)

        results = []
        for i, res in enumerate(reranking.results, 1):
            og_index = res[0] 
            results.append({
                "rank": i,
                "confidence": 1 - fused_results[og_index]['$dist'],
                "doc_id": fused_results[og_index]['original_doc_id'],
                "chunk_index": fused_results[og_index]['chunk_index'],
                "chunk_text": fused_results[og_index]['chunk_text'],
            })
        return results

    def full_text_search(self, keyword_query):
        res_bm25_doc = self.ns.query(
                rank_by=("chunk_text", "BM25", keyword_query[:1023]),
                top_k=5,
                include_attributes=["chunk_text","original_doc_id","chunk_index"],
            )
        results = []
        for i, row in enumerate(res_bm25_doc.rows, 1):
                    row_dict = row.model_dump()
                    results.append({
                        "rank": i,
                        "confidence": 1 - row_dict['$dist'],
                        "doc_id": row_dict['original_doc_id'],
                        "chunk_index": row_dict['chunk_index'],
                        "chunk_text": row_dict['chunk_text'],
                    })
        return results

    def typesense_search(self, keyword_query):
        res = self.ts_client.collections[self.name_space].documents.search({
            "q": keyword_query,
            "query_by": "displayName,markdown,notesMarkdown",
            "page": 1,
            "per_page": 5,
            # Some params expect string enums per API docs; either works in recent clients,
            # but this is the safest form:
            "prefix": "false",
        })
        results = []
        for i, hit in enumerate(res["hits"], 1):
            highlight_section = hit.get("highlight") or hit.get("highlights")
            snippet = None
            if highlight_section:
                snippet_data = highlight_section.get("markdown") or {}
                snippet = snippet_data.get("snippet")
            results.append({
                "rank": i,
                "confidence": hit["text_match"],
                "doc_id": hit["document"]["id"],
                "chunk_index": -1,
                "chunk_text": snippet or "",
            })
        return results
    
    def reciprocal_rank_fusion(self, result_lists, k=60):
        """
        Reciprocal Rank Fusion (RRF) algorithm for combining multiple ranked result lists.
        
        Args:
            result_lists: List of ranked result lists
            k: Parameter that controls the contribution of lower-ranked results (default: 60)
        
        Returns:
            List of results sorted by RRF score in descending order
        """
        scores = {}
        all_results = {}
        
        for results in result_lists:
            for rank in range(1, len(results) + 1):
                item = results[rank - 1]
                item_id = item['id']
                scores[item_id] = scores.get(item_id, 0) + 1.0 / (k + rank)
                all_results[item_id] = item
        
        # Sort by score in descending order and return results
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Add the score to each result and return
        for doc_id, score in sorted_items:
            all_results[doc_id]['dist'] = score
        
        return [all_results[doc_id] for doc_id, _ in sorted_items]

# DEPRECATED
# ------------------------------------------------------------
def compare_searches(ns):
    voy = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    # Load one of your saved chunk files to sample true chunk_text from disk
    # paths = sorted(glob.glob("all_chunks.pkl"))
    # assert paths, "No saved chunk files found (all_chunks.pkl)."
    # chunks = pickle.load(open(paths[-1], "rb"))
    # sample = random.choice([c for c in chunks if "chunk_text" in c and c.get("original_doc_id")])

    samples = ns.query(
        rank_by=("id", "asc"),
        include_attributes=True,
        top_k=500
    )
    sample = random.choice(samples.rows)

    chunk_text = sample["chunk_text"]
    doc_id = sample["original_doc_id"]
    chunk_index = sample["chunk_index"]
    # print(chunk_text)


    print("Test for document: ", doc_id, "chunk_index: ", chunk_index)
    print("First 100 chars of chunk text: ", chunk_text[:100])

    # Vector search using the chunk's text as the query
    qe = voy.contextualized_embed(inputs=[[chunk_text]], model="voyage-context-3", input_type="query").results[0].embeddings[0]
    res_vec = ns.query(
        rank_by=("vector", "ANN", qe),
        top_k=5,
        include_attributes=True,
    )
    print("--------------------------------")
    print("\nVector search (global):")
    for obj in res_vec:
        if obj[0] == 'rows':
            for i, row in enumerate(obj[1], 1):
                row_dict = row.model_dump()
                print(i, 1 - row_dict['$dist'], row_dict['original_doc_id'], row_dict['chunk_index'])
                if i == 1:
                    print(row_dict['chunk_text'][:300])

    res_vec = ns.multi_query(
        queries=[
          {
            "rank_by": ("vector", "ANN", qe),
            "top_k": 5,
            "include_attributes": True,
          },
            {
              "rank_by": ("chunk_text", "BM25", chunk_text[:1023]),
              "top_k": 5,
              "include_attributes": True,
            }
        ]
      )
    print("--------------------------------")
    print("hybrid search (global):")
    fused_results = reciprocal_rank_fusion([res_vec.results[0].rows, res_vec.results[1].rows])
    for i, obj in enumerate(fused_results[:5], 1):
        row_dict = obj.model_dump()
        print(i, 1 - row_dict['$dist'], row_dict['original_doc_id'], row_dict['chunk_index'])
        if i == 1:
            print(row_dict['chunk_text'][:300])


    # BM25 should also bring the same chunk to the top within the doc
    res_bm25_doc = ns.query(
        rank_by=("chunk_text", "BM25", chunk_text[:1023]),
        top_k=5,
        include_attributes=["chunk_text","original_doc_id","chunk_index"],
    )
    print("\nBM25:")
    for obj in res_bm25_doc:
        if obj[0] == 'rows':
            for i, row in enumerate(obj[1], 1):
                row_dict = row.model_dump()
                print(i, 1 - row_dict['$dist'], row_dict['original_doc_id'], row_dict['chunk_index'])
                if i == 1:
                    print(row_dict['chunk_text'][:300])

    # for obj in res_vec:
    #     if obj[0] == 'results':
    #         for i, row in enumerate(obj[1], 1):
    #             row_dict = row.model_dump()
    #             breakpoint()
    #             print(i, 1 - row_dict['$dist'], row_dict['original_doc_id'], row_dict['chunk_index'])
                # print(row_dict['chunk_text'][:100])


    # Expectation: row from same doc with same chunk_index appears at rank 1 (or within top-3).
    # Doc-scoped variant (should be rank 1 = exact chunk)
    res_vec_doc = ns.query(
        rank_by=("vector", "ANN", qe),
        filters=("original_doc_id", "Eq", doc_id),
        top_k=5,
        include_attributes=["chunk_text","original_doc_id","chunk_index"],
    )
    print("\nVector search (doc-scoped):")
    for obj in res_vec_doc:
        if obj[0] == 'rows':
            for i, row in enumerate(obj[1], 1):
                row_dict = row.model_dump()
                print(i, 1 - row_dict['$dist'], row_dict['original_doc_id'], row_dict['chunk_index'])

def verify_chunk_text(ns, chunk_index):
    test_doc_id = "2f67c110-0055-4417-9d83-768ca77ca59d"
    test_chunk_index = chunk_index

    paths = sorted(glob.glob("all_chunks_20250812_193405.pkl"))
    assert paths, "No saved chunk files found (all_chunks_20250812_193405.pkl)."
    chunks = pickle.load(open(paths[-1], "rb"))
    chunk_text = chunks[chunk_index]["chunk_text"]

    # Get the actual stored chunk
    results = ns.query(
        filters=('And', (("original_doc_id", "Eq", test_doc_id), ("chunk_index", "Eq", test_chunk_index))),
        include_attributes=["chunk_text", "original_doc_id", "chunk_index"],
        top_k=1
    )

    print("chunk_text:", chunk_text)
    print("pickl text:", results.rows[0].chunk_text)
    print("diff:", [c for c in chunk_text if c not in results.rows[0].chunk_text])

def export_to_csv(q_and_a_eval, n_samples):
    with open(f'q_and_a_eval_{n_samples}.csv', 'w') as f:
        writer = csv.writer(f)
        # 'typesense_results', 'vector_search_results', 'hybrid_search_results', 'full_text_search_results'
        writer.writerow(['question', 'doc_id', 'chunk_text (first 250 chars)', 'correct', 'correct_to_the_chunk'])
        for k, v in q_and_a_eval.items():
            writer.writerow([v['question'], v['doc_id'], v['chunk_text'][:250], v['correct'], v['chunk_correct']])


# verify_chunk_text(ns, 5)
# compare_searches(ns)
n_samples = 100
search_client = SearchFunctions()
# generate_questions_from_chunk(n_samples=n_samples)
pickle_path = f"sample_questions_{n_samples}.pkl"
with open(pickle_path, 'rb') as f:
    sample_questions = pickle.load(f)
ask_questions(sample_questions, n_samples=n_samples)

pickle_path = f"questions_and_answers_{n_samples}.pkl"
with open(pickle_path, 'rb') as f:
    q_and_a = pickle.load(f)
evaluate_results(q_and_a, n_samples)

pickle_path = f"q_and_a_eval_{n_samples}.pkl"
with open(pickle_path, 'rb') as f:
    q_and_a = pickle.load(f)
export_to_csv(q_and_a, n_samples)