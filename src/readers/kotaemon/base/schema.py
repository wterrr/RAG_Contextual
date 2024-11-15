from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, Union, List, Dict
from langchain.schema.messages import AIMessage as LCAIMessage
from langchain.schema.messages import HumanMessage as LCHumanMessage
from langchain.schema.messages import SystemMessage as LCSystemMessage
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import Document as BaseDocument

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

IO_Type = TypeVar("IO_Type", "Document", str)
SAMPLE_TEXT = "A sample Document from kotaemon"

class Document(BaseDocument):
    content: Any = None
    source: Optional[str] = None
    channel: Optional[Literal["chat", "info", "index", "debug", "plot"]] = None

    def __init__(self, content: Optional[Any] = None, *args, **kwargs):
        if content is None:
            kwargs["content"] = kwargs.get("text") or kwargs.get("embedding", "<EMBEDDING>")
        elif isinstance(content, Document):
            kwargs.update(content.dict(), **kwargs)
        else:
            kwargs["content"] = content
            kwargs["text"] = str(content) if content else ""
        super().__init__(*args, **kwargs)

    def __bool__(self) -> bool:
        return bool(self.content)

    @classmethod
    def example(cls) -> Document:
        return cls(text=SAMPLE_TEXT, metadata={"filename": "README.md", "category": "codebase"})

    def __str__(self) -> str:
        return str(self.content)

class DocumentWithEmbedding(Document):
    """
    Subclass of Document that enforces the presence of an embedding attribute.
    """
    def __init__(self, embedding: List[float], *args, **kwargs):
        kwargs["embedding"] = embedding
        super().__init__(*args, **kwargs)

class BaseMessage(Document):
    def __add__(self, other: Any):
        raise NotImplementedError("Addition not implemented for BaseMessage")

    def to_openai_format(self) -> ChatCompletionMessageParam:
        raise NotImplementedError("Method to_openai_format must be implemented in subclasses")

class SystemMessage(BaseMessage):
    def __init__(self, content: str, base_message: BaseMessage, lc_system_message: LCSystemMessage):
        self.base_message = base_message
        self.lc_system_message = lc_system_message
        self.content = content

    def to_openai_format(self) -> ChatCompletionMessageParam:
        return {"role": "system", "content": self.content}

class AIMessage(BaseMessage):
    def __init__(self, content: str, base_message: BaseMessage, lc_ai_message: LCAIMessage):
        self.base_message = base_message
        self.lc_ai_message = lc_ai_message
        self.content = content

    def to_openai_format(self) -> ChatCompletionMessageParam:
        return {"role": "assistant", "content": self.content}

class HumanMessage(BaseMessage):
    def __init__(self, content: str, base_message: BaseMessage, lc_human_message: LCHumanMessage):
        self.base_message = base_message
        self.lc_human_message = lc_human_message
        self.content = content

    def to_openai_format(self) -> ChatCompletionMessageParam:
        return {"role": "user", "content": self.content}

class RetrievedDocument(Document):
    """
    Document subclass with retrieval-related information such as score and metadata.
    """
    score: float = Field(default=0.0)
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)

class LLMInterface(AIMessage):
    candidates: List[str] = Field(default_factory=list)
    completion_tokens: int = -1
    total_tokens: int = -1
    prompt_tokens: int = -1
    total_cost: float = 0.0
    logits: List[List[float]] = Field(default_factory=list)
    messages: List[AIMessage] = Field(default_factory=list)
    logprobs: List[float] = Field(default_factory=list)

class ExtractorOutput(Document):
    """
    Represents the output of an extractor.
    """
    matches: List[str]
