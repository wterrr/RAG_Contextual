import os
import logging
from icecream import ic
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from starlette.responses import StreamingResponse, Response
from src.tools.contextual_rag_tool import load_contextual_rag_tool
from src.constants import SERVICE, TEMPERATURE, MODEL_ID, STREAM, AGENT_TYPE

load_dotenv(override=True)

class ChatbotAssistant:
    query_engine: AgentRunner
    tools_dict: dict
    
    def __init__(self):
        self.tools = self.load_tools()
        self.query_engine = self.create_query_engine()
        
    def load_tools(self):
        contextual_rag_tool = load_contextual_rag_tool()
        return contextual_rag_tool
    
    def add_tools(self, tools: FunctionTool | list[FunctionTool]) -> None:
        if isinstance(tools, FunctionTool):
            tools = [tools]
        self.tools.extend(tools)
        ic(f"Add: {len(tools)} tools.")
        self.query_engine = (
            self.create_query_engine()
        )
        
    def create_query_engine(self):
        llm = self.load_model(SERVICE, MODEL_ID)
        Settings.llm = llm
        
        ic(AGENT_TYPE, len(self.tools))
        
        if AGENT_TYPE == "openai":
            query_engine = OpenAIAgent.from_tools(
                tools=self.tools, verbose=True, llm=llm
            )
        else:
            raise ValueError("Unknown agent type")
        
        return query_engine
    
    def load_model(self, service, model_id):
        logging.info(f"Logging model: {model_id}")
        logging.info("This action can take a few minutes!")
        
        if service=="ollama":
            logging.info(f"Loading Ollama model: {model_id}")
            return Ollama(model=model_id, temperature=TEMPERATURE)
        elif service=="openai":
            logging.info(f"Loading OpenAI model: {model_id}")
            return OpenAI(model=model_id, temperature=TEMPERATURE, api_key=os.getenv("OPENAI_API_KEY"))
        elif service=="gemini":
            logging.info(f"Loading Gemini model: {model_id}")
            return Gemini(model=model_id, temperature=TEMPERATURE, api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise NotImplementedError("The implementation for other types of LLMs are not ready yet!")
        
    def complete(self, query):
        return self.query_engine.chat(query)
    
    def predict(self, prompt):
        if STREAM:
            streaming_response = self.query_engine.stream_chat(prompt)
            
            return StreamingResponse(
                streaming_response.response_gen,
                media_type="application/text, charset=utf-8"
            )
        else:
            return Response(
                self.query_engine.chat(prompt).reponse,
                media_type="application/text, charset=utf-8"
            )
        