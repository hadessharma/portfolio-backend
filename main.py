import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="Portfolio Chat API")

# Configure CORS
origins = [
    "https://sharmacodes.com",
    "https://hadessharma.github.io",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Knowledge Base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
kb_path = os.path.join(BASE_DIR, "knowledge_base.md")

try:
    with open(kb_path, "r", encoding="utf-8") as f:
        knowledge_base_content = f.read()
except FileNotFoundError:
    knowledge_base_content = "Knowledge base not found."

# Configure Gemini API
# Note: GEMINI_API_KEY must be set in Vercel Environment Variables
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Initialize Generative Model
model = genai.GenerativeModel(
    model_name="gemini-3.1-flash-lite-preview",
    system_instruction=(
        "You are an AI assistant representing Deep Sharma. "
        "Your target audience is technical recruiters. "
        "Keep your responses strictly under 3 sentences or a maximum of 3 short bullet points unless explicitly asked for more detail. "
        "Keep the tone professional, objective, and direct. Do not use conversational filler. "
        "Base your answers ONLY on the provided knowledge base. If the information is not in the knowledge base, do not make assumptions. "
        "Instead, state directly that you do not have that specific information, but they can reach out to Deep directly. "
        "Always use Markdown formatting for readability. Format links as clickable Markdown links (e.g., [LinkedIn](url)) and use bullet points for lists and contact details.\n\n"
        f"Knowledge Base:\n{knowledge_base_content}"
    ),
    generation_config=genai.types.GenerationConfig(
        temperature=0.2,
        top_p=0.8,
        top_k=40,
        max_output_tokens=200,
    )
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def generate():
        try:
            response = await model.generate_content_async(request.message, stream=True)
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error generating response: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "FastAPI is running on Vercel!"}

@app.get("/models")
def list_models():
    models = []
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            models.append(m.name)
    return {"models": models}
