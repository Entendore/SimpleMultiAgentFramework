# app.py

import asyncio
import json
import os
import re
import uuid
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import Cookie, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

import uvicorn
from contextlib import asynccontextmanager
import psutil
import platform
from collections import OrderedDict

# --- Config ---
MEMORY_DIR = "./chat_memory"
AGENTS_FILE = "./agents.json"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

AVAILABLE_MODELS: List[str] = []
LLM_MODEL: str = "phi3"
EMBEDDING_MODEL: str = "nomic-embed-text"

os.makedirs(MEMORY_DIR, exist_ok=True)

# --- Global locks ---
_ollama_start_lock = asyncio.Lock()
_global_store = None

# --- Model Metadata ---
MODEL_METADATA = {
    "phi3": {
        "name": "Phi-3",
        "description": "Fast, compact, great for reasoning",
        "tags": ["fast", "reasoning", "small"],
        "speed": "⚡⚡⚡⚡☆",
        "quality": "⭐⭐⭐⭐☆",
        "context": "128K tokens",
        "size": "3.8B",
        "memory_usage": {
            "ram_min": 4,
            "ram_recommended": 6,
            "vram": 2.5,
            "disk": 2.1
        },
        "recommended_for": ["math", "code"]
    },
    "stablelm": {
        "name": "StableLM",
        "description": "Balanced performance, good for general use",
        "tags": ["balanced", "general"],
        "speed": "⚡⚡⚡☆☆",
        "quality": "⭐⭐⭐⭐☆",
        "context": "8K tokens",
        "size": "1.6B",
        "memory_usage": {
            "ram_min": 2,
            "ram_recommended": 4,
            "vram": 1.2,
            "disk": 0.9
        },
        "recommended_for": ["writing", "general"]
    },
    "llama3": {
        "name": "Llama 3",
        "description": "High quality, versatile, from Meta",
        "tags": ["high-quality", "versatile"],
        "speed": "⚡⚡☆☆☆",
        "quality": "⭐⭐⭐⭐⭐",
        "context": "8K tokens",
        "size": "8B",
        "memory_usage": {
            "ram_min": 8,
            "ram_recommended": 12,
            "vram": 5.0,
            "disk": 4.7
        },
        "recommended_for": ["writing", "meme", "zoomer", "boomer"]
    },
    "gemma2": {
        "name": "Gemma 2",
        "description": "Google's lightweight, efficient model",
        "tags": ["fast", "small", "efficient"],
        "speed": "⚡⚡⚡⚡☆",
        "quality": "⭐⭐⭐☆☆",
        "context": "8K tokens",
        "size": "2B",
        "memory_usage": {
            "ram_min": 3,
            "ram_recommended": 5,
            "vram": 1.8,
            "disk": 1.2
        },
        "recommended_for": ["code", "math"]
    },
    "mistral": {
        "name": "Mistral",
        "description": "Strong reasoning, good balance",
        "tags": ["reasoning", "balanced"],
        "speed": "⚡⚡⚡☆☆",
        "quality": "⭐⭐⭐⭐☆",
        "context": "32K tokens",
        "size": "7B",
        "memory_usage": {
            "ram_min": 7,
            "ram_recommended": 10,
            "vram": 4.5,
            "disk": 4.2
        },
        "recommended_for": ["math", "code", "writing"]
    },
    "qwen2": {
        "name": "Qwen 2",
        "description": "Alibaba's capable multilingual model",
        "tags": ["multilingual", "balanced"],
        "speed": "⚡⚡⚡☆☆",
        "quality": "⭐⭐⭐⭐☆",
        "context": "32K tokens",
        "size": "7B",
        "memory_usage": {
            "ram_min": 7,
            "ram_recommended": 10,
            "vram": 4.3,
            "disk": 4.0
        },
        "recommended_for": ["writing", "general"]
    }
}

# --- Model Manager ---
class ModelManager:
    def __init__(self, max_models_in_memory=2):
        self.max_models_in_memory = max_models_in_memory
        self.loaded_models = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get_agent_for_model(self, agent_key: str, agent_config: dict):
        async with self._lock:
            model_name = agent_config["model"]
            
            if model_name in self.loaded_models:
                self.loaded_models.move_to_end(model_name)
                if agent_key in self.loaded_models[model_name]:
                    return self.loaded_models[model_name][agent_key]
            
            if len(self.loaded_models) >= self.max_models_in_memory:
                oldest_model, _ = self.loaded_models.popitem(last=False)
                print(f"🔄 Swapping out model: {oldest_model}")
            
            agent = DynamicAgent(agent_key, agent_config)
            
            if model_name not in self.loaded_models:
                self.loaded_models[model_name] = {}
            self.loaded_models[model_name][agent_key] = agent
            
            print(f"✅ Loaded model: {model_name} (total: {len(self.loaded_models)})")
            return agent

model_manager = ModelManager(max_models_in_memory=2)

# --- Ollama Helpers ---
async def is_ollama_running() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/")
            return r.status_code == 200
    except:
        return False

async def ensure_ollama_running():
    if await is_ollama_running():
        return True

    async with _ollama_start_lock:
        if await is_ollama_running():
            return True

        print("⚠️ Ollama not running. Attempting to start Ollama...")

        try:
            if os.name == "nt":
                subprocess.Popen(["start", "ollama"], shell=True)
            else:
                subprocess.Popen(["ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"❌ Failed to launch Ollama automatically: {e}")
            return False

        for _ in range(15):
            if await is_ollama_running():
                print("✅ Ollama started successfully.")
                return True
            await asyncio.sleep(1)

        print("❌ Ollama did not start after 15 seconds.")
        return False

async def get_ollama_models() -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code == 200:
                data = r.json()
                models = []
                for model in data.get("models", []):
                    name = model["name"]
                    if ":" in name:
                        name = name.split(":")[0]
                    models.append(name)
                return sorted(set(models))
    except Exception as e:
        print(f"Error fetching models: {e}")
    return []

def find_best_model(available: List[str], candidates: List[str]) -> str:
    for c in candidates:
        if c in available:
            return c
    return available[0] if available else "unknown"

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    selected_agent: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    agent_name: str
    agent_key: str

class AgentCreateRequest(BaseModel):
    key: Optional[str] = None  # ← Made optional
    name: str
    task: str
    system_prompt: str
    model: str
    transitions: List[str] = Field(default_factory=list)

# --- Load Agents ---
def load_agents():
    if os.path.exists(AGENTS_FILE):
        with open(AGENTS_FILE) as f:
            return json.load(f)
    return {}

def save_agents(agents):
    with open(AGENTS_FILE, "w") as f:
        json.dump(agents, f, indent=2)

agents_config: Dict = load_agents()
sessions: Dict[str, "SessionMemory"] = {}

# --- Helpers ---
CHAIN_PATTERN = re.compile(r"\[CHAIN:([a-zA-Z0-9_,]+)\]", re.IGNORECASE)

def parse_agent_chain(message: str):
    match = CHAIN_PATTERN.search(message)
    if match:
        chain_str = match.group(1)
        agents = [a.strip().lower() for a in chain_str.split(",") if a.strip()]
        clean_msg = CHAIN_PATTERN.sub("", message).strip()
        return clean_msg, agents
    return message, None

def extract_answer(response: str) -> str:
    return response.strip() or "[No response]"

# --- Dynamic Agent ---
class DynamicAgent:
    def __init__(self, key: str, config: dict):
        self.key = key
        self.name = config["name"]
        self.task = config["task"]
        self.model = config.get("model", LLM_MODEL)

        self.llm = ChatOllama(model=self.model, temperature=0.7)
        self.prompt = ChatPromptTemplate.from_template(
            config["system_prompt"] + "\n\n"
            "Relevant context:\n{retrieved_context}\n\n"
            "Chat history:\n{history}\n\nUser: {input}\n{agent_name}:"
        )

    async def run(self, message: str, history: str, retrieved_context: str = "") -> str:
        if not await ensure_ollama_running():
            return "[System] Ollama failed to start. Please start it manually."
        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({
                "input": message,
                "history": history,
                "retrieved_context": retrieved_context,
                "agent_name": self.name
            })
            return response.content.strip()
        except Exception as e:
            return f"[Error: {str(e)}]"

async def get_agent(key: str) -> Optional["DynamicAgent"]:
    if key not in agents_config:
        return None
    return await model_manager.get_agent_for_model(key, agents_config[key])

# --- Session Memory ---
class SessionMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[str] = []
        try:
            self.vector_store = Chroma(
                collection_name=f"session_{session_id}",
                embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
                persist_directory=MEMORY_DIR
            )
        except Exception as e:
            print(f"⚠️ Failed to init vector store for {session_id}: {e}")
            self.vector_store = None

    def add_message(self, role: str, content: str):
        text = f"{role}: {content}"
        self.history.append(text)
        if self.vector_store:
            ts = datetime.now().isoformat()
            self.vector_store.add_documents([
                Document(page_content=text, metadata={"session": self.session_id, "ts": ts})
            ])
        add_to_global_memory(text, self.session_id)

    def get_history(self, max_lines: int = 8) -> str:
        return "\n".join(self.history[-max_lines:])

    def search(self, query: str, k: int = 5) -> List[dict]:
        if not self.vector_store:
            return []
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [{"content": doc.page_content, "score": score} for doc, score in results]
        except Exception as e:
            print(f"Search error: {e}")
            return []

def get_or_create_session(session_id: str) -> SessionMemory:
    if session_id not in sessions:
        sessions[session_id] = SessionMemory(session_id)
    return sessions[session_id]

# --- Global Memory ---
def add_to_global_memory(content: str, session_id: str):
    global _global_store
    if _global_store is None:
        try:
            _global_store = Chroma(
                collection_name="global_memory",
                embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
                persist_directory=MEMORY_DIR
            )
        except Exception as e:
            print(f"⚠️ Global memory init failed: {e}")
            return
    try:
        ts = datetime.now().isoformat()
        _global_store.add_documents([
            Document(page_content=content, metadata={"session": session_id, "ts": ts})
        ])
    except Exception as e:
        print(f"⚠️ Failed to add to global memory: {e}")

# --- Agent Routing ---
def route_to_agent(message: str) -> str:
    msg_lower = message.lower().strip()

    pirate_keywords = ["pirate", "arrr", "matey", "captain", "ship", "treasure", "booty", "nautical"]
    zoomer_keywords = ["zoomer", "gen z", "rizz", "no cap", "fr", "bet", "gyat", "skibidi", "sigma"]
    boomer_keywords = ["boomer", "back in my day", "kids these days", "rotary phone", "vinyl"]
    meme_keywords = ["meme", "funny", "caption", "distracted boyfriend", "drake", "viral"]
    
    math_keywords = [
        "math", "calculate", "compute", "solve", "equation", "arithmetic",
        "what is", "how much", "+", "-", "*", "/", "=", "reverse", "step by step", "algorithm"
    ]
    code_keywords = ["code", "python", "debug", "error", "function", "syntax"]
    writing_keywords = ["write", "essay", "story", "poem", "creative", "describe"]

    scores = {
        "math": 0, "code": 0, "writing": 0,
        "pirate": 0, "zoomer": 0, "boomer": 0, "meme": 0
    }

    for word in pirate_keywords:
        if word in msg_lower:
            scores["pirate"] += 3
    for word in zoomer_keywords:
        if word in msg_lower:
            scores["zoomer"] += 3
    for word in boomer_keywords:
        if word in msg_lower:
            scores["boomer"] += 3
    for word in meme_keywords:
        if word in msg_lower:
            scores["meme"] += 3

    for word in math_keywords:
        if word in msg_lower:
            scores["math"] += 1
    for word in code_keywords:
        if word in msg_lower:
            scores["code"] += 1
    for word in writing_keywords:
        if word in msg_lower:
            scores["writing"] += 1

    if any(char in msg_lower for char in "+-*/=") and any(c.isdigit() for c in msg_lower):
        scores["math"] += 2

    if " like a " in msg_lower or " as a " in msg_lower:
        if "pirate" in msg_lower:
            scores["pirate"] += 4
        elif "zoomer" in msg_lower or "gen z" in msg_lower:
            scores["zoomer"] += 4
        elif "boomer" in msg_lower:
            scores["boomer"] += 4

    best_agent = max(scores, key=scores.get)
    return best_agent if scores[best_agent] > 0 else "math"

# --- FastAPI App ---
templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_system()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Initialize System ---
async def initialize_system():
    global AVAILABLE_MODELS, LLM_MODEL, EMBEDDING_MODEL, agents_config, _global_store, model_manager

    await ensure_ollama_running()
    AVAILABLE_MODELS = await get_ollama_models()
    LLM_MODEL = find_best_model(AVAILABLE_MODELS, ["phi3", "llama3", "gemma2"])
    EMBEDDING_MODEL = find_best_model(AVAILABLE_MODELS, ["nomic-embed-text"])

    try:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if ram_gb < 6:
            max_models = 1
        elif ram_gb < 12:
            max_models = 2
        elif ram_gb < 24:
            max_models = 3
        else:
            max_models = 4
        model_manager.max_models_in_memory = max_models
        print(f"🧠 Auto-configured model capacity: {max_models} models (RAM: {ram_gb:.1f}GB)")
    except:
        model_manager.max_models_in_memory = 2

    try:
        _global_store = Chroma(
            collection_name="global_memory",
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=MEMORY_DIR
        )
    except Exception as e:
        print(f"⚠️ Global memory init failed: {e}")

    # Always ensure all 7 core agents exist
    core_agents = {
        "math": {
            "name": "Math Tutor",
            "task": "Solve math and algorithmic problems",
            "system_prompt": "You are a patient math and computational thinking tutor. Explain step-by-step.",
            "model": LLM_MODEL,
            "transitions": ["code"]
        },
        "code": {
            "name": "Code Assistant",
            "task": "Help with programming",
            "system_prompt": "You are a helpful, precise coding assistant.",
            "model": LLM_MODEL,
            "transitions": ["writing", "math"]
        },
        "writing": {
            "name": "Writing Coach",
            "task": "Assist with creative writing",
            "system_prompt": "You are a thoughtful creative writing coach.",
            "model": LLM_MODEL,
            "transitions": ["code", "math"]
        },
        "pirate": {
            "name": "Captain Blackbeard",
            "task": "Speak like a pirate",
            "system_prompt": "Ye be a fearsome pirate! Use 'arrr!', 'matey', 'shiver me timbers'. Never break character!",
            "model": LLM_MODEL,
            "transitions": ["writing"]
        },
        "zoomer": {
            "name": "Gen-Z Pal",
            "task": "Talk like a Zoomer",
            "system_prompt": "You're a chill Gen-Z friend. Use slang like 'rizz', 'no cap', 'fr', 'bet'. Keep it playful!",
            "model": LLM_MODEL,
            "transitions": ["writing"]
        },
        "meme": {
            "name": "Meme Lord",
            "task": "Generate meme captions",
            "system_prompt": "Generate responses in meme formats like 'Distracted Boyfriend' or 'Drake'. Be hilarious and concise.",
            "model": LLM_MODEL,
            "transitions": ["zoomer", "writing"]
        },
        "boomer": {
            "name": "Old-School Boomer",
            "task": "Speak like a Baby Boomer",
            "system_prompt": "You are a classic Baby Boomer. Say 'Back in my day...' and 'Kids these days...'. Be nostalgic and slightly grumpy.",
            "model": LLM_MODEL,
            "transitions": ["writing", "math"]
        }
    }

    # Merge core agents with existing config (preserves custom agents)
    for key, agent in core_agents.items():
        if key not in agents_config:
            agents_config[key] = agent

    save_agents(agents_config)

# --- Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "ollama_running": await is_ollama_running()}

@app.get("/models")
async def get_models():
    return AVAILABLE_MODELS

@app.get("/models/info")
async def get_model_info():
    available = set(AVAILABLE_MODELS)
    filtered = {k: v for k, v in MODEL_METADATA.items() if k in available}
    
    for model in AVAILABLE_MODELS:
        if model not in MODEL_METADATA:
            filtered[model] = {
                "name": model,
                "description": "Unknown model",
                "tags": ["unknown"],
                "speed": "❓",
                "quality": "❓",
                "context": "❓",
                "size": "❓",
                "memory_usage": None,
                "recommended_for": []
            }
    
    return filtered

@app.get("/system/info")
async def get_system_info():
    try:
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage('/')
        return {
            "os": platform.system().lower(),
            "ram_total_gb": round(ram.total / (1024 ** 3), 1),
            "ram_available_gb": round(ram.available / (1024 ** 3), 1),
            "swap_total_gb": round(swap.total / (1024 ** 3), 1),
            "swap_used_gb": round(swap.used / (1024 ** 3), 1),
            "swap_percent": swap.percent,
            "disk_free_gb": round(disk.free / (1024 ** 3), 1),
            "cpu_count": psutil.cpu_count()
        }
    except Exception as e:
        return {
            "os": platform.system().lower(),
            "ram_total_gb": None,
            "ram_available_gb": None,
            "swap_total_gb": None,
            "swap_used_gb": None,
            "swap_percent": None,
            "disk_free_gb": None,
            "cpu_count": None
        }

@app.get("/models/status")
async def get_model_status():
    try:
        ram = psutil.virtual_memory()
        return {
            "loaded_models": list(model_manager.loaded_models.keys()),
            "max_models": model_manager.max_models_in_memory,
            "ram_total_gb": round(ram.total / (1024 ** 3), 1),
            "ram_available_gb": round(ram.available / (1024 ** 3), 1),
            "ram_percent_used": ram.percent
        }
    except Exception as e:
        return {
            "loaded_models": list(model_manager.loaded_models.keys()),
            "max_models": model_manager.max_models_in_memory,
            "ram_total_gb": None,
            "ram_available_gb": None,
            "ram_percent_used": None
        }

@app.get("/agents")
async def list_agents():
    return agents_config

@app.post("/agents")
async def create_agent(req: AgentCreateRequest):
    if not req.key or not req.name or not req.system_prompt:
        raise HTTPException(status_code=400, detail="Missing required fields")
    if req.key in agents_config:
        raise HTTPException(status_code=400, detail="Agent key already exists")
    agents_config[req.key] = {
        "name": req.name,
        "task": req.task,
        "system_prompt": req.system_prompt,
        "model": req.model,
        "transitions": req.transitions,
    }
    save_agents(agents_config)
    return {"status": "created"}

@app.put("/agents/{agent_key}")
async def update_agent(agent_key: str, req: AgentCreateRequest):
    """Fixed: Removed reference to _agent_instances and made key optional"""
    if agent_key not in agents_config:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agents_config[agent_key] = {
        "name": req.name,
        "task": req.task,
        "system_prompt": req.system_prompt,
        "model": req.model,
        "transitions": req.transitions,
    }
    save_agents(agents_config)
    # The model manager will automatically handle the updated configuration
    return {"status": "updated"}

@app.get("/agents/dfa")
async def get_dfa():
    nodes = []
    edges = []
    for key, cfg in agents_config.items():
        nodes.append({"id": key, "label": cfg["name"], "task": cfg["task"]})
        for target in cfg.get("transitions", []):
            if target in agents_config:
                edges.append({"from": key, "to": target})
    return {"nodes": nodes, "edges": edges}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID")
    session = get_or_create_session(session_id)
    
    clean_msg, agent_chain = parse_agent_chain(req.message)
    
    if agent_chain:
        invalid = [a for a in agent_chain if a not in agents_config]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Unknown agent(s): {', '.join(invalid)}")
        
        current_input = clean_msg
        final_agent_key = agent_chain[-1]
        
        for i, agent_key in enumerate(agent_chain):
            agent = await get_agent(agent_key)
            if i == 0:
                history = session.get_history()
                retrieved = session.search(clean_msg)
                context = "\n".join([r["content"] for r in retrieved]) if retrieved else ""
                response = await agent.run(current_input, history, context)
            else:
                rephrase_prompt = f"Rephrase this in your style:\n\n{current_input}\n\nRephrased:"
                response = await agent.run(rephrase_prompt, "", "")
            current_input = extract_answer(response)
        
        final_response = current_input
        final_agent = await get_agent(final_agent_key)
        agent_name = final_agent.name if final_agent else final_agent_key
        
    else:
        agent_key = req.selected_agent or route_to_agent(clean_msg)
        agent = await get_agent(agent_key)
        if not agent:
            agent_key = "math"
            agent = await get_agent(agent_key)
        
        history = session.get_history()
        retrieved = session.search(clean_msg)
        context = "\n".join([r["content"] for r in retrieved]) if retrieved else ""
        response = await agent.run(clean_msg, history, context)
        final_response = extract_answer(response)
        agent_name = agent.name
        final_agent_key = agent_key

    session.add_message("User", req.message)
    session.add_message(agent_name, final_response)
    
    return ChatResponse(
        message=final_response,
        agent_name=agent_name,
        agent_key=final_agent_key
    )

@app.get("/session")
async def get_session(session_id: str = Cookie(None)):
    if not session_id or session_id not in sessions:
        return {"history": []}
    return {"history": sessions[session_id].history}

@app.post("/clear")
async def clear_session(response: Response, session_id: str = Cookie(None)):
    if session_id and session_id in sessions:
        del sessions[session_id]
    response.delete_cookie("session_id")
    return {"status": "cleared"}

@app.get("/search/current")
async def search_current(q: str, session_id: str = Cookie(None)):
    if not session_id or session_id not in sessions:
        return {"results": []}
    return {"results": sessions[session_id].search(q)}

@app.get("/search/global")
async def search_global(q: str):
    if _global_store is None:
        return {"results": []}
    try:
        results = _global_store.similarity_search_with_score(q, k=5)
        return {
            "results": [
                {
                    "content": doc.page_content,
                    "score": score,
                    "session": doc.metadata.get("session"),
                    "ts": doc.metadata.get("ts")
                }
                for doc, score in results
            ]
        }
    except Exception as e:
        print(f"Global search error: {e}")
        return {"results": []}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, session_id: str = Cookie(None)):
    if not session_id:
        session_id = str(uuid.uuid4())
    response = templates.TemplateResponse("index.html", {"request": request})
    response.set_cookie(key="session_id", value=session_id)
    return response

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)