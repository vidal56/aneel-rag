"""
serve_gaia.py — Servidor vLLM para CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it

Expõe API compatível com OpenAI em http://0.0.0.0:8080/v1
Compatível com: openai Python SDK, LiteLLM, qualquer cliente OpenAI-compatible

Uso:
    python3 serve_gaia.py
"""
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "/models/gaia-4b")
MODEL_NAME  = os.getenv("MODEL_NAME", "gaia-4b")
HOST        = os.getenv("HOST", "0.0.0.0")
PORT        = os.getenv("PORT", "8080")
GPU_MEM     = os.getenv("GPU_MEMORY_UTILIZATION", "0.70")
MAX_LEN     = os.getenv("MAX_MODEL_LEN", "8192")
DTYPE       = os.getenv("DTYPE", "bfloat16")   # L4 (Ada Lovelace) suporta bfloat16
QUANT       = os.getenv("QUANTIZATION", "")

# ── Auto-download se o modelo não existir ────────────────────────────────────
# Necessário no Docker: o volume /models começa vazio na primeira execução.
if not Path(MODEL_PATH).exists():
    print(f"Modelo não encontrado em {MODEL_PATH} — baixando (~8GB, aguarde)...")
    from huggingface_hub import snapshot_download
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
        local_dir=MODEL_PATH,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print(f"Modelo salvo em: {MODEL_PATH}")

print("=" * 60)
print(" ANEEL RAG — Servidor GAIA")
print("=" * 60)
print(f" Modelo  : {MODEL_PATH}")
print(f" Endpoint: http://{HOST}:{PORT}/v1")
print(f" GPU mem : {GPU_MEM} (70% de 24GB L4 = ~16.8GB)")
print(f" Max len : {MAX_LEN} tokens")
print(f" dtype   : {DTYPE}")
print(f" Quant   : {QUANT if QUANT else 'nenhuma'}")
print("=" * 60)

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model",                   MODEL_PATH,
    "--served-model-name",       MODEL_NAME,
    "--host",                    HOST,
    "--port",                    PORT,
    "--gpu-memory-utilization",  GPU_MEM,
    "--max-model-len",           MAX_LEN,
    "--dtype",                   DTYPE,
    "--trust-remote-code",
    "--enable-prefix-caching",
]

if QUANT and QUANT.lower() != "none":
    cmd += ["--quantization", QUANT]

print(f"\nIniciando vLLM...\n")
subprocess.run(cmd)
