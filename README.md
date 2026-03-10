# AI Agent Speech-to-Text, Text-to-Speech, Web Search

FastAPI-based AI assistant with:
- Text chat
- Voice input (speech-to-text)
- Voice output (text-to-speech)
- Web search grounding using Serper API
- Frontend chat UI with chat history

## Project Structure

- `main.py` — FastAPI routes (`/chat`, `/voice-chat`)
- `agent_core.py` — STT, TTS, web search, answer generation
- `static/index.html` — frontend UI
- `requirements.txt` — Python dependencies

## Requirements

- Python 3.10+
- macOS/Linux/Windows
- Internet access (for Serper search and model downloads)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
pip install sounddevice scipy openai-whisper edge-tts langchain-huggingface python-multipart accelerate
python -m spacy download en_core_web_sm
```

## Environment Variables

Set your Serper API key:

```bash
export SERPER_API_KEY="your_serper_api_key"
```

## Run

```bash
python main.py
```

Open:
- `http://127.0.0.1:8000`

## API Endpoints

- `POST /chat`
  - Body: `{"query": "your question"}`
  - Response: `{"response": "..."}`

- `POST /voice-chat`
  - Form-data: audio file field name `file`
  - Response: `{"query_text": "...", "response": "..."}`

## Notes

- Typed input returns text response (no auto speech playback).
- Mic input uses speech-to-speech flow (voice in, voice + text out).
- If FAISS is not installed, the app safely falls back without vector DB.
