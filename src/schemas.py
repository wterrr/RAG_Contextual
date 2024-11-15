from llama_index.core.bridge.pydantic import BaseModel

class RAGType(BaseModel):
    ORIGINAL = "origin"
    CONTEXTUAL = "contextual"
    BOTH = "both"

class DocumentMetaData(BaseModel):
    doc_id: str
    original_content: str
    contextualized_content: str

class ElasticSearch(BaseModel):
    doc_id: str
    content: str
    contextualized_content: str
    score: float