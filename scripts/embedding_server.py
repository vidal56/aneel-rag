"""
embedding_server.py — FastAPI wrapping BGE-M3 + BGE-reranker-v2-m3

Expõe:
  GET  /health   → healthcheck
  POST /embed    → dense + sparse vectors (BGE-M3 fp16)
  POST /rerank   → scores normalizados (BGE-reranker-v2-m3 fp16)

Consumido pelo rag_tools.py via HTTP (evita carregar GPU no processo do ADK).
~25% da VRAM L4 (~6GB): BGE-M3 fp16 ≈ 2.2GB + reranker ≈ 600MB + overhead.
"""

import os
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("embedding")

BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH", "BAAI/bge-m3")

app = FastAPI(title="ANEEL Embedding Service", version="1.0.0")

# ── Singletons (carregados no startup) ────────────────────────────────────────

_bge = None
_reranker = None


def _get_bge():
    global _bge
    if _bge is None:
        from FlagEmbedding import BGEM3FlagModel
        log.info(f"Carregando BGE-M3 de {BGE_MODEL_PATH} ...")
        _bge = BGEM3FlagModel(BGE_MODEL_PATH, use_fp16=True)
        log.info("BGE-M3 pronto.")
    return _bge


def _get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        log.info("Carregando BGE-reranker-v2-m3 ...")
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        log.info("Reranker pronto.")
    return _reranker


# ── Schemas ───────────────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]


class RerankRequest(BaseModel):
    query: str
    passages: List[str]


class RerankResponse(BaseModel):
    scores: List[float]


# ── Startup: pré-carrega modelos ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    # Limita este processo a 25% da VRAM da L4 (24GB → ~6GB).
    # Deve ser chamado ANTES de qualquer alocação CUDA.
    # Os outros 70% ficam livres para o container GAIA (vLLM).
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.25, device=0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU limitada a 25% da VRAM: {total_gb * 0.25:.1f}GB de {total_gb:.1f}GB")

    log.info("Inicializando modelos de embedding...")
    _get_bge()
    _get_reranker()
    log.info("Serviço de embedding pronto.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "bge": _bge is not None, "reranker": _reranker is not None}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text não pode ser vazio")
    try:
        model = _get_bge()
        output = model.encode(
            [req.text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"][0].tolist()
        sparse_weights = output["lexical_weights"][0]
        return EmbedResponse(
            dense=dense,
            sparse_indices=[int(k) for k in sparse_weights.keys()],
            sparse_values=[float(v) for v in sparse_weights.values()],
        )
    except Exception as e:
        log.error(f"Erro no embed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if not req.passages:
        return RerankResponse(scores=[])
    try:
        reranker = _get_reranker()
        pairs = [[req.query, p] for p in req.passages]
        raw = reranker.compute_score(pairs, normalize=True)
        scores = raw if isinstance(raw, list) else [float(raw)]
        return RerankResponse(scores=[float(s) for s in scores])
    except Exception as e:
        log.error(f"Erro no rerank: {e}")
        raise HTTPException(status_code=500, detail=str(e))
