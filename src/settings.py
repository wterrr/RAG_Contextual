from llama_index.core.bridge.pydantic import Field, BaseModel

from src.constants import (
    CONTEXTUAL_CHUNK_SIZE,
    CONTEXTUAL_SERVICE,
    CONTEXTUAL_MODEL,
    ORIGINAL_RAG_COLLECTION_NAME,
    CONTEXTUAL_RAG_COLLECTION_NAME,
    QDRANT_URL,
    ELASTIC_SEARCH_URL,
    ELASTIC_SEARCH_INDEX_NAME,
    NUM_CHUNK_TO_RECALL,
    SEMANTIC_WEIGHT,
    BM25_WEIGHT,
    TOP_N,
)

class Settings(BaseModel):
    chunk_size: int = Field(description="Chunk size", default=CONTEXTUAL_CHUNK_SIZE)
    
    service: str = Field(description="LLM service", default=CONTEXTUAL_SERVICE)
    
    model: str = Field(description="LLM model", default=CONTEXTUAL_MODEL)
    
    original_rag_collection_name: str = Field(description="Original RAG collection name", default=ORIGINAL_RAG_COLLECTION_NAME)
    
    contextual_rag_collection_name: str = Field(description="Contextual RAG collection name", default=CONTEXTUAL_RAG_COLLECTION_NAME)
    
    qdrant_url: str = Field(description="QdrantVectorDB URL", default=QDRANT_URL)
    
    elastic_search_url = Field(description="Elastic URL", default=ELASTIC_SEARCH_URL)
    
    elastic_search_index_name = Field(description="Elastic index name", default=ELASTIC_SEARCH_INDEX_NAME)
    
    num_chunks_to_recall = Field(description="Number of chunks to recall", default=NUM_CHUNK_TO_RECALL)
    
    semantic_weight = Field(description="Semantic weight", default=SEMANTIC_WEIGHT)
    
    bm25_weight = Field(description="BM25 weight", default=BM25_WEIGHT)
    
    top_n = Field(description="Top n documents after reranking", default=TOP_N)
    
setting = Settings()