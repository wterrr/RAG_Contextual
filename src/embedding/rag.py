import os
import sys
import json
import uuid
from tqdm import tqdm
from icecream import ic
from pathlib import Path
from typing import Literal
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore, Node
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.response.schema import Response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import (
    Settings,
    Document,
    QueryBundle,
    StorageContext,
    VectorStoreIndex,
)

from src.constants import CONTEXTUAL_PROMPT, QA_PROMPT
from src.schemas import RAGType, DocumentMetaData
from src.readers.file_reader import parse_multiple_files
from src.embedding.elastic_search import ElasticSearch
from src.settings import Settings as ConfigSettings, setting as config_setting

def time_format():
    now = datetime.now()
    return f'{now.strftime("%H:%M:%S")} - DEBUG - '

load_dotenv()
ic.configureOutput(includeContext=True, prefix=time_format)
Settings.chunk_size = config_setting.chunk_size

class RAG:
    """
    Class to handle indexing and searching for both Origin and Contextual RAG.
    """

    def __init__(self, setting: ConfigSettings):
        self.setting = setting
        ic(setting)

        embed_model = OpenAIEmbedding()
        Settings.embed_model = embed_model

        self.llm = OpenAI(model=setting.model)
        Settings.llm = self.llm

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )

        self.es = ElasticSearch(
            url=setting.elastic_search_url, index_name=setting.elastic_search_index_name
        )

        self.qdrant_client = QdrantClient(url=setting.qdrant_url)

    def split_document(self, document: Document | list[Document], show_progress=True) -> list[list[Document]]:
        if isinstance(document, Document):
            document = [document]
        documents = []

        for doc in tqdm(document, desc="Splitting...") if show_progress else document:
            nodes = self.splitter.get_nodes_from_documents([doc])
            documents.append([Document(text=node.get_content()) for node in nodes])

        return documents

    def add_contextual_content(self, origin_document: Document, splited_documents: list[Document]) -> list[Document]:
        whole_document = origin_document.text
        documents, documents_metadata = [], []

        for chunk in splited_documents:
            messages = [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content=CONTEXTUAL_PROMPT.format(
                    WHOLE_DOCUMENT=whole_document, CHUNK_CONTENT=chunk.text
                )),
            ]
            response = self.llm.chat(messages)
            contextualized_content = response.message.content
            new_chunk = contextualized_content + "\n\n" + chunk.text
            doc_id = str(uuid.uuid4())

            documents.append(Document(text=new_chunk, metadata={"doc_id": doc_id}))
            documents_metadata.append(DocumentMetaData(
                doc_id=doc_id, original_content=whole_document, contextualized_content=contextualized_content
            ))

        return documents, documents_metadata

    def get_contextual_documents(self, raw_documents: list[Document], splited_documents: list[list[Document]]) -> tuple[list[Document], list[DocumentMetaData]]:
        documents, documents_metadata = [], []
        for raw_document, splited_document in tqdm(
            zip(raw_documents, splited_documents), desc="Adding contextual content ...", total=len(raw_documents)
        ):
            document, metadata = self.add_contextual_content(raw_document, splited_document)
            documents.extend(document)
            documents_metadata.extend(metadata)
        return documents, documents_metadata

    def ingest_data(self, documents: list[Document], show_progress=True, type="contextual"):
        collection_name = self.setting.contextual_rag_collection_name if type == "contextual" else self.setting.original_rag_collection_name
        ic(type, collection_name)

        vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=show_progress)

    def insert_data(self, documents: list[Document], show_progress=True, type="contextual"):
        collection_name = self.setting.contextual_rag_collection_name if type == "contextual" else self.setting.original_rag_collection_name
        ic(type, collection_name)

        vector_store_index = self.get_qdrant_vector_store_index(client=self.qdrant_client, collection_name=collection_name)
        for document in tqdm(documents, desc=f"Adding more data to {type} ...") if show_progress else documents:
            vector_store_index.insert(document)

    def get_qdrant_vector_store_index(self, client: QdrantClient, collection_name: str) -> VectorStoreIndex:
        ic(collection_name)
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

    def get_query_engine(self, type="contextual") -> BaseQueryEngine | dict[str, BaseQueryEngine]:
        ic(type)
        if type == RAGType.ORIGIN:
            return self.get_qdrant_vector_store_index(client=self.qdrant_client, collection_name=self.setting.original_rag_collection_name).as_query_engine()
        elif type == RAGType.CONTEXTUAL:
            return self.get_qdrant_vector_store_index(client=self.qdrant_client, collection_name=self.setting.contextual_rag_collection_name).as_query_engine()
        elif type == RAGType.BOTH:
            return {
                "origin": self.get_qdrant_vector_store_index(client=self.qdrant_client, collection_name=self.setting.original_rag_collection_name).as_query_engine(),
                "contextual": self.get_qdrant_vector_store_index(client=self.qdrant_client, collection_name=self.setting.contextual_rag_collection_name).as_query_engine(),
            }

    def run_ingest(self, folder_dir: str | Path, type="contextual") -> None:
        ic(folder_dir, type)
        raw_documents = parse_multiple_files(folder_dir)
        splited_documents = self.split_document(raw_documents)
        ingest_documents = [doc for each_splited in splited_documents for doc in each_splited]

        if type in [RAGType.ORIGIN, RAGType.BOTH]:
            self.ingest_data(ingest_documents, type=RAGType.ORIGIN)

        if type in [RAGType.CONTEXTUAL, RAGType.BOTH]:
            contextual_documents, contextual_documents_metadata = self.get_contextual_documents(raw_documents=raw_documents, splited_documents=splited_documents)
            self.ingest_data(contextual_documents, type=RAGType.CONTEXTUAL)
            self.es.index_documents(contextual_documents_metadata)
            ic(f"Ingested data for {type}")

    def run_add_files(self, files_or_folders: list[str], type="contextual"):
        ic(files_or_folders, type)
        raw_documents = parse_multiple_files(files_or_folders)
        splited_documents = self.split_document(raw_documents)
        ingest_documents = [doc for each_splited in splited_documents for doc in each_splited]

        if type in [RAGType.ORIGIN, RAGType.BOTH]:
            self.insert_data(ingest_documents, type=RAGType.ORIGIN)

        if type in [RAGType.CONTEXTUAL, RAGType.BOTH]:
            contextual_documents, contextual_documents_metadata = self.get_contextual_documents(raw_documents=raw_documents, splited_documents=splited_documents)
            self.insert_data(contextual_documents, type=RAGType.CONTEXTUAL)
            self.es.index_documents(contextual_documents_metadata)
            ic(f"Added files for {type}")
            
    def origin_rag_search(self, query: str) -> str:
        ic(query)

        index = self.get_query_engine(RAGType.ORIGIN)
        return index.query(query)

    # Compute score according to: https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
    def contextual_rag_search(self, query: str, k: int = 150, debug: bool = False) -> str:
        ic(query, k, debug)

        # Weights for combining semantic and BM25 scores
        semantic_weight = self.setting.semantic_weight
        bm25_weight = self.setting.bm25_weight

        # Retrieve semantic results
        index = self.get_qdrant_vector_store_index(
            self.qdrant_client, self.setting.contextual_rag_collection_name
        )
        retriever = VectorIndexRetriever(index=index, similarity_top_k=k)
        query_engine = RetrieverQueryEngine(retriever=retriever)
        semantic_results: Response = query_engine.query(query)

        # Collect document IDs and content from semantic results
        semantic_doc_id = [node.metadata["doc_id"] for node in semantic_results.source_nodes]
        def get_content_by_doc_id(doc_id: str) -> str:
            for node in semantic_results.source_nodes:
                if node.metadata["doc_id"] == doc_id:
                    return node.text
            return ""

        # Retrieve BM25 results
        bm25_results = self.es.search(query, k=k)
        bm25_doc_id = [result.doc_id for result in bm25_results]

        # Combine document IDs and calculate scores
        combined_nodes = []
        combined_ids = set(semantic_doc_id + bm25_doc_id)
        semantic_count = bm25_count = both_count = 0

        for doc_id in combined_ids:
            score, content = 0, ""

            # Score based on semantic ranking
            if doc_id in semantic_doc_id:
                index = semantic_doc_id.index(doc_id)
                score += semantic_weight / (index + 1)
                content = get_content_by_doc_id(doc_id)
                semantic_count += 1

            # Score based on BM25 ranking
            if doc_id in bm25_doc_id:
                index = bm25_doc_id.index(doc_id)
                score += bm25_weight / (index + 1)
                if not content:
                    content = f"{bm25_results[index].contextualized_content}\n\n{bm25_results[index].content}"
                bm25_count += 1
            if doc_id in semantic_doc_id and doc_id in bm25_doc_id:
                both_count += 1

            combined_nodes.append(NodeWithScore(node=Node(text=content), score=score))

        if debug:
            ic(semantic_count, bm25_count, both_count)

        # Rerank combined results
        reranker = CohereRerank(top_n=self.setting.top_n, api_key=os.getenv("COHERE_API_KEY"))
        query_bundle = QueryBundle(query_str=query)
        retrieved_nodes = reranker.postprocess_nodes(combined_nodes, query_bundle)

        # Format response with context
        contexts = [n.node.text for n in retrieved_nodes]
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content=QA_PROMPT.format(context_str=json.dumps(contexts), query_str=query)),
        ]

        return self.llm.chat(messages).message.content