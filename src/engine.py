from loguru import logger
from .prompts import mqr_prompt
from functools import lru_cache
from qdrant_client.http import models 
from qdrant_client import QdrantClient
from .utils import StrictLegalQueryParser
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from .config import DEVICE, EMBED_MODEL_ID, CACHE_DIR, HF_TOKEN, QDRANT_URL, QDRANT_API_KEY


VECTOR_STORE = None

@lru_cache(maxsize=1)
def load_embeddings():
    """Caches the heavy transformer model in memory."""
    logger.info("📡 Loading SentenceTransformer to RAM...")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID, 
        model_kwargs={'device': DEVICE, 'token': HF_TOKEN},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}, 
        cache_folder=str(CACHE_DIR / "huggingface")
    )

@lru_cache(maxsize=1)
def get_reranker():
    """Caches FlashRank model to avoid reloading on every query."""
    logger.info("🚀 Warming up FlashRank: ms-marco-MiniLM-L-12-v2")
    return FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=10)
    
def get_vector_store():
    global VECTOR_STORE
    if VECTOR_STORE: return VECTOR_STORE

    logger.info("🛠️  Initializing Hybrid Vector Engine...")

    try:
        embeddings = load_embeddings()

        logger.debug(f"✅  Dense Embeddings loaded on {DEVICE}")

        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25", cache_dir=str(CACHE_DIR / "fastembed"))

        logger.debug("✅  Sparse BM25 Embeddings ready")

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)        
        collection_name = "indian_legal_library"
            
        # Audit existing collections
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if not exists:
            logger.warning(f"⚠️  Collection '{collection_name}' not found. Creating schema...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                sparse_vectors_config={"langchain-sparse": SparseVectorParams(index={"on_disk": True})}
            )

            for field in ["metadata.law_name", "metadata.section"]:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            
            logger.success(f"🆕  Schema created with Payload Indexing for '{collection_name}'")
        else:
            info = client.get_collection(collection_name)
            logger.info(f"💡  Collection '{collection_name}' active with {info.points_count} points")

        VECTOR_STORE = QdrantVectorStore(
            client=client, 
            collection_name=collection_name, 
            embedding=embeddings,
            sparse_embedding=sparse_embeddings, 
            sparse_vector_name="langchain-sparse", 
            retrieval_mode=RetrievalMode.HYBRID
        )
        return VECTOR_STORE
    except Exception as e:
        logger.critical(f"🛑  Failed to initialize Vector Store: {e}")
        raise

def get_retriever(llm_for_queries, law_name_filter=None):
    vs = get_vector_store()

    logger.debug("🔄  Configuring Multi-Query Expansion & FlashRank Reranker...")

    MULTI_QUERY_PROMPT = mqr_prompt()

    search_kwargs={"k": 10, "fetch_k": 30, 'lambda_mult': 0.7}

    if law_name_filter:
        logger.info(f"🎯 Applying Payload Filter: {law_name_filter}")
        if isinstance(law_name_filter, list):
            match_logic = models.MatchAny(any=law_name_filter)
        else:
            match_logic = models.MatchValue(value=law_name_filter)
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.law_name", # Must match your index key
                    match=match_logic
                )
            ]
        )

    base_ret = vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    query_chain = MULTI_QUERY_PROMPT | llm_for_queries | StrictLegalQueryParser()

    mq_ret = MultiQueryRetriever(retriever=base_ret, llm_chain=query_chain, parser_key='lines')

    logger.debug("🚀  Loading FlashRank: ms-marco-MiniLM-L-12-v2")
    compressor = get_reranker()
    
    logger.success("🚀  Retrieval Engine: Multi-Query + FlashRank Reranker is ready.")
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=mq_ret)