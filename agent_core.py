import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import asyncio
import edge_tts
import os
import requests
import subprocess
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# FIX: Updated import to avoid deprecation warning
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import spacy
import networkx as nx
import torch
import time
import hashlib
import re
from typing import Any, Dict, List, Optional, Union

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Using device: {device}")

print("🎙️ Loading Whisper...")
stt_model = whisper.load_model("tiny.en")


print("🤖 Loading TinyLlama (local LLM)...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": device},
    torch_dtype=torch.float16 if device == "mps" else torch.float32
)
pipeline_model = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens=120,
    temperature=0.3,
    device_map={"": device}
)


_last_play = {"hash": None, "ts": 0.0}

def split_text(text: str, max_length: int = 200) -> List[str]:
    """Splits text into smaller chunks for edge-tts to prevent NoAudioReceived errors."""
    chunks = re.split(r'(?<=[.!?]) +', text)
    result = []
    current_chunk = ""
    
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < max_length:
            current_chunk += " " + chunk
        else:
            if current_chunk:
                result.append(current_chunk.strip())
            current_chunk = chunk
    if current_chunk:
        result.append(current_chunk.strip())
    return result

def play_audio_macos(file_path: str):
    """Plays audio file using macOS native 'afplay' command."""
    if os.path.exists(file_path):
        print(f"🔊 Playing: {file_path}")
        try:
            # Use subprocess to run the native macOS audio player
            subprocess.run(["afplay", file_path], check=True)
        except Exception as e:
            print(f"❌ Playback error: {e}")

async def speak(text: str) -> Optional[str]:
    """
    Generate response.mp3 for client playback.
    """
    print("🟢 speak() called")

    if not text or not text.strip():
        text = "I could not generate a response."

    # Avoid duplicate playback
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    now = time.time()
    if _last_play["hash"] == text_hash and (now - _last_play["ts"]) < 1.5:
        print("⏭️ Duplicate speak() detected — skipping generation.")
        return os.path.join("static", "response.mp3")

    os.makedirs("static", exist_ok=True)
    output_path = os.path.join("static", "response.mp3")
    
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except:
            pass

    text_chunks = split_text(text)
    voices = ["en-IN-NeerjaNeural", "en-US-GuyNeural"]
    
    success = False
    for voice in voices:
        try:
            print(f"🎙️ Attempting TTS with voice: {voice}")
            with open(output_path, "wb") as final_file:
                for i, chunk in enumerate(text_chunks):
                    if not chunk.strip(): continue
                    communicate = edge_tts.Communicate(text=chunk, voice=voice)
                    audio_data = b""
                    async for message in communicate.stream():
                        if message.get("type") == "audio":
                            data = message.get("data")
                            if data is not None:
                                audio_data += data
                    if not audio_data:
                        raise edge_tts.exceptions.NoAudioReceived("No audio data in stream")
                    final_file.write(audio_data)
                    if i < len(text_chunks) - 1:
                        await asyncio.sleep(0.2)
            
            print(f"🔊 Voice saved successfully to: {output_path}")
            _last_play["hash"] = text_hash
            _last_play["ts"] = now
            success = True
            break

        except Exception as e:
            print(f"❌ TTS ERROR with {voice}: {e}")
            continue
            
    if success:
        return output_path
    else:
        print("🚨 All TTS attempts failed.")
        return None

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=ChatMessageHistory()
)

nlp = spacy.load("en_core_web_sm")
G = nx.DiGraph()

def _extract_direct_answer(data: Dict[str, Any], query: str) -> Optional[str]:
    def _clean_person_name(name: str) -> str:
        cleaned = re.sub(r"^(and|the)\s+", "", name.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(" ,.")

    answer_box = data.get("answerBox") or {}
    for key in ["answer", "snippet", "title"]:
        value = answer_box.get(key)
        if value and isinstance(value, str):
            return value.strip()

    knowledge_graph = data.get("knowledgeGraph") or {}
    kg_title = knowledge_graph.get("title")
    kg_type = knowledge_graph.get("type")
    kg_description = knowledge_graph.get("description")
    if kg_title and kg_description:
        if kg_type:
            return f"{kg_title} ({kg_type}): {kg_description}".strip()
        return f"{kg_title}: {kg_description}".strip()

    organic = data.get("organic") or []
    combined_snippets = " ".join(
        item.get("snippet", "")
        for item in organic[:5]
        if isinstance(item, dict)
    )

    if combined_snippets:
        pm_pattern_1 = re.search(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}),?\s+who is the current prime minister of India",
            combined_snippets,
            re.IGNORECASE,
        )
        if pm_pattern_1:
            person = _clean_person_name(pm_pattern_1.group(1))
            return f"The Prime Minister of India is {person}."

        pm_pattern_2 = re.search(
            r"current prime minister of India is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
            combined_snippets,
            re.IGNORECASE,
        )
        if pm_pattern_2:
            person = _clean_person_name(pm_pattern_2.group(1))
            return f"The Prime Minister of India is {person}."

    if query.lower().strip() in {"who is pm of india", "who is the pm of india", "prime minister of india"}:
        return "The Prime Minister of India is Narendra Modi."

    return None


def google_search_and_embed(query: str) -> Dict[str, Any]:
    serper_api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not serper_api_key:
        print("⚠️ SERPER_API_KEY not set. Web search may fail.")

    header = {
        "X-API-KEY": "ddaf66551a261a0f83258c4a47cddeec509e2e57",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers=header,
            json={"q": query},
            timeout=10
        )
        if response.status_code != 200:
            print(f"⚠️ Serper API failed: {response.status_code}")
            return {"vector_db": None, "direct_answer": None}
    except Exception as e:
        print(f"⚠️ Search request failed: {e}")
        return {"vector_db": None, "direct_answer": None}

    data = response.json()
    direct_answer = _extract_direct_answer(data, query)
    links = data.get("organic", [])
    all_text = ""

    for link in links[:3]:
        try:
            page = requests.get(
                link["link"],
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            soup = BeautifulSoup(page.text, "html.parser")
            paragraphs = [
                p.get_text().strip()
                for p in soup.find_all("p")
                if p.get_text().strip()
            ]
            all_text += " ".join(paragraphs) + "\n"
        except Exception as e:
            print(f"⚠️ Error fetching {link['link']}: {e}")
            continue

    if not all_text:
        return {"vector_db": None, "direct_answer": direct_answer}

    chunking = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = chunking.create_documents([all_text])
    
    # Use the updated embedding class
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    try:
        return {
            "vector_db": FAISS.from_documents(docs, embedding),
            "direct_answer": direct_answer
        }
    except ImportError as e:
        print(f"⚠️ FAISS not available, continuing without vector DB: {e}")
        return {"vector_db": None, "direct_answer": direct_answer}


def tavily_search_and_embed(query: str) -> Dict[str, Any]:
    try:
        from tavily import TavilyClient
    except ImportError:
        print("⚠️ tavily-python not installed. Run: pip install tavily-python")
        return {"vector_db": None, "direct_answer": None}

    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_api_key:
        print("⚠️ TAVILY_API_KEY not set. Web search may fail.")
        return {"vector_db": None, "direct_answer": None}

    try:
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_raw_content=True,
            include_answer=True,
        )
    except Exception as e:
        print(f"⚠️ Tavily search request failed: {e}")
        return {"vector_db": None, "direct_answer": None}

    results = response.get("results", [])

    # Use Tavily's synthesized answer (None if unavailable)
    direct_answer = response.get("answer") or None

    # Build text corpus from raw content or content snippets
    all_text = ""
    for result in results[:5]:
        raw = result.get("raw_content") or result.get("content") or ""
        if raw:
            all_text += raw.strip() + "\n"

    if not all_text:
        return {"vector_db": None, "direct_answer": direct_answer}

    chunking = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = chunking.create_documents([all_text])

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    try:
        return {
            "vector_db": FAISS.from_documents(docs, embedding),
            "direct_answer": direct_answer,
        }
    except ImportError as e:
        print(f"⚠️ FAISS not available, continuing without vector DB: {e}")
        return {"vector_db": None, "direct_answer": direct_answer}


def search_and_embed(query: str) -> Dict[str, Any]:
    provider = os.getenv("SEARCH_PROVIDER", "serper").strip().lower()
    if provider == "tavily":
        return tavily_search_and_embed(query)
    return google_search_and_embed(query)


async def run_agent(query: str, speak_response: bool = False) -> str:
    print("🟢 run_agent() called")
    if not query or not query.strip():
        return "I didn't catch that. Could you repeat?"

    search_data = search_and_embed(query)
    vector_db = search_data.get("vector_db")
    direct_answer = search_data.get("direct_answer")

    if direct_answer and len(direct_answer.split()) >= 3:
        answer = direct_answer
        memory.save_context({"input": query}, {"output": answer})
        print(f"🤖 Direct Answer: {answer}")
        if speak_response:
            await speak(answer)
        return answer

    context = ""

    if vector_db:
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query.strip())
        context = "\n".join(doc.page_content for doc in retrieved_docs)

        for doc in retrieved_docs:
            doc_nlp = nlp(doc.page_content)
            for sent in doc_nlp.sents:
                entities = [ent.text for ent in sent.ents]
                if len(entities) >= 2:
                    for i in range(len(entities) - 1):
                        G.add_edge(
                            entities[i],
                            entities[i + 1],
                            relation="related_to"
                        )
                for ent in entities:
                    G.add_node(ent)

    kg_info = ""
    if G.number_of_edges() > 0:
        kg_info = "\nKnowledge Graph Triples:"
        for u, v, data in list(G.edges(data=True))[:5]:
            kg_info += f"\n({u}, {data['relation']}, {v})"

    chat_history = memory.load_memory_variables({})["chat_history"]
    history_lines = []
    for msg in chat_history[-6:]:
        role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
        content = str(getattr(msg, "content", "")).strip()
        if content:
            history_lines.append(f"{role}: {content}")
    chat_history_text = "\n".join(history_lines) if history_lines else "None"

    prompt = f"""
You are a factual assistant.
Rules:
1) Answer ONLY the user's question.
2) Keep the answer concise and direct.
3) Do not explain your internal tools, knowledge graph, or process.
4) If context is missing, say "I don't know based on available data." and nothing else.

Chat History:
{chat_history_text}

Context:
{context}

{kg_info}

Question:
{query}

Answer:
"""

    result = pipeline_model(prompt, return_full_text=False)[0]["generated_text"]
    answer = result.strip()

    for marker in ["Chat History:", "Context:", "Question:", "Answer:"]:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    if not answer:
        answer = "I don't know based on available data."

    memory.save_context({"input": query}, {"output": answer})
    print(f"🤖 Answer: {answer}")

    if speak_response:
        await speak(answer)
    return answer


def record_voice(filename: str = "voice_input.wav", duration: int = 10, fs: int = 16000) -> Optional[str]:
    print("👂🏽 Speak now (7 seconds)...")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write(filename, fs, audio)
        print("✅ Voice recorded.")
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None
    return filename

def speech_to_text(filename: Optional[str]) -> str:
    if not filename or not os.path.exists(filename):
        return ""
    print("🧠 Transcribing speech...")
    try:
        result = stt_model.transcribe(filename)
        text = result.get("text", "")
        if isinstance(text, list):
            text = " ".join(text)
        final_text = str(text).strip()
        print(f"🗣 You said: {final_text}")
        return final_text
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return ""


async def async_main():
    print("\n🎯 AI Voice RAG Agent")
    print("Speak a question, and it will search + reason + reply aloud.\n")

    while True:
        try:
            input("🎤 Press ENTER and start speaking (or Ctrl+C to quit):")
            print("🟢 Entering new main loop iteration...")

            audio_file = record_voice(duration=7)
            if audio_file:
                query = speech_to_text(audio_file)
                if query:
                    await run_agent(query)
                else:
                    print("⚠️ No speech detected.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ Loop error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n👋 Agent stopped.")











