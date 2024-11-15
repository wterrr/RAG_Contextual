from fastapi import APIRouter, Request
from .service import ChatbotAssistant
from src.tools.booking import load_booking_tools

router = APIRouter()
assistant = ChatbotAssistant()

assistant.add_tools(loading_booking_tools())

@router.post("/complete")
async def complete_text(request: Request):
    data = await request.json()
    message = data.get("message")
    response = assistant.predict(message)
    return response