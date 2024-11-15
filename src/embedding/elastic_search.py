from icecream import ic
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from llama_index.core.bridge.pydantic import Field
from src.schemas import DocumentMetaData, ElasticSearch as ElasticSearchResponse

class ElasticSearch:
    url = Field(..., description="ElasticSearch URL")
    
    def __init__(self, url, index_name):
        ic(url, index_name)
        
        self.es_client = Elasticsearch(url)
        self.index_name = index_name
        self.create_index()
    
    # cfg refrences https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-create-index.html
    def create_index(self):
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False,
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "text", "index": False},
                }
            },
        }
        
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            ic(f"Created index: {self.index_name}")
    
    # cfg refrences https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html#search-api-response-body        
    def index_document(self, documents_metadata: list[DocumentMetaData]):
        ic("Indexing document...")
        
        actions = [
            {
                "_index": self.index_name,
                "source": {
                    "doc_id": metadata.doc_id,
                    "content": metadata.original_content,
                    "contextualized_content": metadata.contextualized_content,
                }
            }
            for metadata in documents_metadata
        ]
        
        success, _ = bulk(self.es_client, actions)
        if success:
            ic("Indexed documents successfully!")
        self.es_client.indices.refresh(index=self.index_name)
        
    def search(self, query, top_k=20):
        ic(query, top_k)
        
        self.es_client.indices.refresh(index=self.index_name)
        
        search = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fileds": ["content", "contextualized_content"],
                }
            },
            "size": top_k,
        }
        
        response = self.es_client.search(index=self.index_name, body=search)
        
        return [
            ElasticSearchResponse(
                doc_id=hit["_source"]["doc_id"],
                content=hit["source"]["content"],
                contextualized_content=hit["source"]["contextualized_content"],
                score=hit["_score"],
            )
            for hit in response["hits"]["hits"]
        ]
        
        