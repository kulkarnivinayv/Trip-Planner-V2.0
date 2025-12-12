# main.py (FASTAPI) — single model load here

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
from dotenv import load_dotenv
import os
import traceback
import groq
from io import BytesIO
from gtts import gTTS
from deep_translator import GoogleTranslator

# Optionally tiktoken for token counting
try:
    import tiktoken
except Exception:
    tiktoken = None

# local project imports (keep as you had)
from agent.agentic_workflow import GraphBuilder  # ensure this is available and not loading extra models
from utils.save_to_document import save_document

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    chat_history: str = ""

class ChatMessage(BaseModel):
    message: str

def count_tokens(text: str) -> int:
    """Approximate token counter (uses tiktoken if available, else fallback)."""
    if not text:
        return 0
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    # fallback rough estimate (1 token ≈ 4 chars)
    return max(1, int(len(text) / 4))

class GroqEmbeddingModel:
    """
    Lightweight cloud-based embedding model using Groq.
    This avoids local model loading (fast startup, Render friendly).
    """
    def encode(self, text):
        try:
            from groq import Groq
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


            response = client.embeddings.create(
                model="text-embedding-3-small",  # light, fast, high-quality
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            print("Groq embedding error:", e)
            return []
        
print("Using Groq Embedding API (text-embedding-3-small)")
model = GroqEmbeddingModel()


class InsuranceQuoteRequest(BaseModel):
    destination: str
    trip_start: str
    trip_end: str
    age: int
    travelers: int = 1
    coverage_type: str = "standard"


@app.post("/insurance/squaremouth")
def insurance_squaremouth(req: InsuranceQuoteRequest):
    """
    Mock Squaremouth API response for testing.
    Replace with real API integration later.
    """

    sample_plans = [
        {
            "provider": "Allianz Travel",
            "plan_name": "Standard International",
            "price": 3499.00,
            "medical": "₹4,00,000",
            "trip_cancellation": "₹1,20,000",
            "baggage": "₹80,000",
            "evacuation": "₹3,00,000"
        },
        {
            "provider": "TATA AIG",
            "plan_name": "Travel Guard",
            "price": 4299.00,
            "medical": "₹7,50,000",
            "trip_cancellation": "₹2,00,000",
            "baggage": "₹1,00,000",
            "evacuation": "₹5,00,000"
        },
        {
            "provider": "HDFC ERGO",
            "plan_name": "Travel Safe",
            "price": 3899.00,
            "medical": "₹10,00,000",
            "trip_cancellation": "₹2,50,000",
            "baggage": "₹90,000",
            "evacuation": "₹7,00,000"
        }
    ]


    return {
        "insurance_quotes": {
            "plans": sample_plans
        }
    }

# -----------------------------
# Translator Request Model
# -----------------------------
class TranslationRequest(BaseModel):
    text: str
    target_language: str

@app.post("/translate")
def translate_text(req: TranslationRequest):
    """
    Translate text and return audio.
    """
    try:
        # 1. Translation
        translated = GoogleTranslator(
            source="auto", target=req.target_language
        ).translate(req.text)

        # 2. Text-to-Speech
        try:
            tts = gTTS(translated, lang=req.target_language)
            audio_buf = BytesIO()
            tts.write_to_fp(audio_buf)
            audio_buf.seek(0)
            audio_bytes = list(audio_buf.read())
        except Exception:
            audio_bytes = None

        return {
            "translated_text": translated,
            "audio_bytes": audio_bytes,
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/chat")
def chat(request: ChatMessage):
    """
    Minimal chat endpoint (keeps things lightweight).
    Currently echoes message with a short confirmation.
    You can extend to return embeddings by using model.encode(...) if needed.
    """
    try:
        msg = request.message or ""
        # if you want embeddings uncomment the next line (embedding arrays are large)
        # emb = model.encode(msg).tolist()
        return {"reply": f"Received: {msg}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_travel_agent(query: QueryRequest):
    try:
        print(f"question={query.question!r}")
        print(f"chat_history={query.chat_history!r}")

        # Build graph (your existing code)
        graph = GraphBuilder(model_provider="groq")
        react_app = graph()

        try:
            png_graph = react_app.get_graph().draw_mermaid_png()
            with open("my_graph.png", "wb") as f:
                f.write(png_graph)
            print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
        except Exception as _:
            # non-fatal if graph rendering fails
            print("Graph rendering failed (continuing):", traceback.format_exc())

        # -----------------------------
        # Build message context
        # -----------------------------
        messages = [
            {"role": "system", "content": "You are a helpful AI travel planner."},
        ]

        if query.chat_history and query.chat_history.strip():
            for line in query.chat_history.split("\n"):
                if line.startswith("user:"):
                    messages.append({
                        "role": "user",
                        "content": line.replace("user:", "").strip()
                    })
                elif line.startswith("assistant:"):
                    messages.append({
                        "role": "assistant",
                        "content": line.replace("assistant:", "").strip()
                    })

        messages.append({"role": "user", "content": query.question})

        print("---- Message Context ----")
        for m in messages:
            print(f"{m['role']}: {m['content']}")
        print("-------------------------")

        # -----------------------------
        # Invoke LLM graph safely
        # -----------------------------
        output = react_app.invoke({"messages": messages})
        print("LLM raw output:", output)

        # -----------------------------
        # Extract final output (robust)
        # -----------------------------
        final_output = ""
        try:
            if isinstance(output, dict) and "messages" in output:
                messages_list = output.get("messages", [])
                if messages_list:
                    last_msg = messages_list[-1]
                    if last_msg is None:
                        final_output = "[No response generated]"
                    elif hasattr(last_msg, "content"):
                        final_output = last_msg.content or "[Empty response]"
                    elif isinstance(last_msg, dict):
                        final_output = last_msg.get("content", "[Missing content]")
                    else:
                        final_output = str(last_msg)
                else:
                    final_output = "[Empty message list]"
            else:
                final_output = str(output) if output else "[Empty output]"
        except Exception as parse_err:
            print("⚠️ Output parsing error:", parse_err)
            final_output = "[Error parsing model output]"

        # -----------------------------
        # Token counting (names match Streamlit)
        # -----------------------------
        prompt_text = "\n".join([m["content"] for m in messages])
        tokens_in = count_tokens(prompt_text)
        tokens_out = count_tokens(final_output)
        total_tokens = tokens_in + tokens_out

        print(f"🧮 Token usage -> Prompt: {tokens_in}, Completion: {tokens_out}, Total: {total_tokens}")
        print("✅ Final extracted output:", final_output)

        return {
            "answer": final_output.strip(),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": total_tokens
        }

    except Exception:
        print("❌ Exception Traceback:\n", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})