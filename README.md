# modeltune

**Local LLM fine-tuning on Apple Silicon (MLX + LoRA) with a search-augmented, OpenAI-compatible serving gateway.**

modeltune covers the full lifecycle of a custom LLM on a single Mac: prepare a dataset, fine-tune an open-source model with LoRA, fuse the adapters, and serve it behind a FastAPI gateway that adds live web search on demand. Any OpenAI-compatible client, such as Open WebUI, can use it as a drop-in backend.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-000000?logo=apple)
![FastAPI](https://img.shields.io/badge/FastAPI-gateway-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9)

---

## Why this project

- **Lower cost and full privacy:** a fine-tuned local model handles domain tasks without paying per token or sending data to external APIs.
- **Fresh knowledge without retraining:** the gateway injects live web search results into the prompt, so the model can answer about current events.
- **Drop-in compatibility:** the gateway speaks the OpenAI API, so existing tools and UIs work without changes.

---

## Highlights

- LoRA fine-tuning with **MLX**, using the Apple Silicon GPU (Metal)
- Adapter fusing into a standalone model for fast serving
- **OpenAI-compatible FastAPI gateway** that proxies requests to `mlx_lm.server`
- `/search` command that triggers **real web search (DuckDuckGo, no API key)** and injects results as context
- Clean `src/` package layout, managed with **uv**, with console entry points
- Containerized gateway with **Docker Compose**

---

## Architecture

```
Open WebUI / any OpenAI client
            │
            ▼
  Gateway (FastAPI, :8100) ──► Web search (DuckDuckGo)
            │
            ▼
  mlx_lm.server (:8099) ──► fine-tuned model (LoRA fused)
```

The gateway passes every request through unchanged, with one exception: if the last user message starts with `/search `, it runs a web search, adds the results to the conversation as context, and forwards the augmented request to the model.

```
/search what's the weather in Lahore today
```

---

## Model

| | |
|---|---|
| **Base model** | `[e.g., Qwen/Qwen3-xB, link to Hugging Face]` |
| **Method** | LoRA (MLX) |
| **Task** | `[what the model was fine-tuned to do]` |
| **Dataset** | `[public dataset name / link, or synthetic sample]` |
| **Fine-tuned weights** | `[link to your Hugging Face model repo]` |

---

## Requirements

- Mac with **Apple Silicon** (M1 or later). MLX requires the Metal GPU.
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker (optional, for running the gateway in a container)

---

## Setup

```bash
git clone https://github.com/devfahadmunir/modeltune.git
cd modeltune

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env   # then edit values if needed
```

---

## 1. Prepare the data

Training data goes in `data/` as JSONL files in the format MLX expects:

```
data/
├── train.jsonl
└── valid.jsonl
```

Dataset preparation logic lives in `src/modeltune/data/`. A small sample is included in `data/sample/` so you can try the pipeline end to end.

---

## 2. Fine-tune with LoRA

Hyperparameters are defined in `lora_config.yaml`.

```bash
mlx_lm.lora --config lora_config.yaml
```

This writes the trained adapters to `./adapters`.

---

## 3. Fuse the adapters

```bash
mlx_lm.fuse \
  --model [BASE_MODEL] \
  --adapter-path ./adapters \
  --save-path ./fused-model
```

---

## 4. Serve the model

### Option A: everything native

```bash
# Terminal 1: model server
mlx_lm.server --model ./fused-model --port 8099

# Terminal 2: gateway
modeltune-gateway
# or: uvicorn modeltune.serving.gateway:app --host 0.0.0.0 --port 8100
```

### Option B: gateway in Docker, model server native (recommended)

The model server **cannot** run in Docker on macOS: Docker runs inside a Linux VM without access to Apple's Metal GPU, which MLX requires. The setup is therefore split by what each component actually needs.

```bash
# Terminal 1: model server on the Mac host
mlx_lm.server --model ./fused-model --port 8099

# Terminal 2: gateway as a container
docker compose up --build
```

The image contains only the gateway and search code (FastAPI, httpx), not MLX. It reaches the native model server through `host.docker.internal`, configured with the `MLX_SERVER_URL` environment variable.

### Connect a client

Point your client at the **gateway (8100)**, not the model server (8099):

| Client location | Base URL |
|---|---|
| Inside Docker (e.g., Open WebUI) | `http://host.docker.internal:8100/v1` |
| Local machine | `http://localhost:8100/v1` |

---

## Results

| Prompt | Base model | Fine-tuned model |
|---|---|---|
| `[example prompt]` | `[base output]` | `[fine-tuned output]` |
| `[example prompt]` | `[base output]` | `[fine-tuned output]` |

`[Add any metrics you measured: accuracy, validation loss, response latency, tokens/sec.]`

---

## Project structure

```
modeltune/
├── src/modeltune/
│   ├── data/            # dataset preparation
│   ├── training/        # LoRA training wrapper
│   ├── serving/
│   │   └── gateway.py   # search-augmented OpenAI-compatible proxy
│   └── search/
│       └── web_search.py  # standalone web search module
├── tests/
├── lora_config.yaml
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Running tests

```bash
pytest
```

---

## Roadmap

- [ ] Streaming (`stream=true`) passthrough in the gateway. Currently the gateway returns a clear `400` error for streaming requests, so configure clients to use non-streaming responses.
- [ ] MCP server exposing web search and model tools
- [ ] Automated evaluation comparing base vs. fine-tuned model
- [ ] Publish fine-tuned weights and model card on Hugging Face

---

## Author

**Fahad Munir**, AI/ML Engineer | Forward Deployed Engineer
[LinkedIn](https://www.linkedin.com/in/devfahadmunir) · [GitHub](https://github.com/devfahadmunir)