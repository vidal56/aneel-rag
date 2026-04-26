"""
rag_tools.py — 6 ferramentas ADK para busca no corpus ANEEL RAG

Stack:
  - BGE-M3 (ONNX fp16) via embedding_server HTTP — dense + sparse
  - Qdrant local (Docker) — collection aneel_rag (281.003 chunks)
  - Busca híbrida RRF top-50 → reranker BGE-reranker top-10
  - Filtros por ano (2016 / 2021 / 2022) e nível (L0 / L1 / L2)

Embedding e reranker rodam no serviço separado (embedding_server.py)
e são consumidos via HTTP — isso isola a VRAM dos dois processos.
"""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

COLLECTION     = os.getenv("QDRANT_COLLECTION", "aneel_rag")
QDRANT_URL     = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_APIKEY  = os.getenv("QDRANT_APIKEY", "")      # vazio = instância local sem auth
EMBEDDING_URL  = os.getenv("EMBEDDING_URL", "http://embedding:8082")

ANOS_VALIDOS   = {"2016", "2021", "2022"}
NIVEIS_VALIDOS = {"L0", "L1", "L2"}

_qdrant = None


def _get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_APIKEY if QDRANT_APIKEY else None,
        )
    return _qdrant


# ── Embedding via HTTP ────────────────────────────────────────────────────────

def _embed(text: str) -> dict:
    """Chama o serviço de embedding e retorna dense + sparse vectors."""
    resp = requests.post(
        f"{EMBEDDING_URL}/embed",
        json={"text": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _rerank(query: str, passages: list[str]) -> list[float]:
    """Chama o serviço de reranking e retorna scores normalizados."""
    resp = requests.post(
        f"{EMBEDDING_URL}/rerank",
        json={"query": query, "passages": passages},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["scores"]


# ── Filtro Qdrant ─────────────────────────────────────────────────────────────

def _build_filter(ano=None, nivel=None, is_active=None, autor=None, assunto=None, tipo_documento=None):
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
    conditions = []
    if ano:
        anos = [a.strip() for a in ano.split(",") if a.strip() in ANOS_VALIDOS]
        if anos:
            conditions.append(FieldCondition(key="ano", match=MatchAny(any=anos)))
    if nivel:
        n = nivel.upper()
        if n in NIVEIS_VALIDOS or n == "METADATA":
            conditions.append(FieldCondition(key="nivel", match=MatchValue(value=n)))
    if is_active is not None:
        conditions.append(FieldCondition(key="is_active", match=MatchValue(value=is_active)))
    if autor:
        conditions.append(FieldCondition(key="autor", match=MatchValue(value=autor)))
    if assunto:
        conditions.append(FieldCondition(key="assunto", match=MatchValue(value=assunto)))
    if tipo_documento:
        conditions.append(FieldCondition(key="tipo_documento", match=MatchValue(value=tipo_documento)))
    return Filter(must=conditions) if conditions else None


# ── Busca híbrida ─────────────────────────────────────────────────────────────

def _hybrid_search(query: str, filt, top_rrf: int = 50, top_k: int = 10) -> list:
    from qdrant_client.models import (
        NamedVector, NamedSparseVector, SparseVector,
        Prefetch, FusionQuery, Fusion,
    )
    client = _get_qdrant()
    emb = _embed(query)
    sparse_vec = SparseVector(
        indices=emb["sparse_indices"],
        values=emb["sparse_values"],
    )
    prefetch = [
        Prefetch(
            query=emb["dense"],
            using="dense",
            limit=top_rrf,
            filter=filt,
        ),
        Prefetch(
            query=sparse_vec,
            using="sparse",
            limit=top_rrf,
            filter=filt,
        ),
    ]
    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_rrf,
        with_payload=True,
    ).points

    # Reranker via HTTP → top_k
    passages = [p.payload.get("text", "") for p in results]
    scores = _rerank(query, passages)
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:top_k]]


def _format_chunks(pontos: list, max_chunks: int = 10) -> str:
    linhas = []
    for i, p in enumerate(pontos[:max_chunks], 1):
        payload = p.payload or {}
        linhas.append(
            f"[DOCUMENTO {i}]\n"
            f"Fonte: {payload.get('doc_id', '?')} | "
            f"Ano: {payload.get('ano', '?')} | "
            f"Nível: {payload.get('nivel', '?')} | "
            f"Score: {p.score:.4f}\n"
            f"{payload.get('text', '')}"
        )
    return "\n\n".join(linhas)


# ── Ferramenta 1 ──────────────────────────────────────────────────────────────

def buscar_legislacao_aneel(
    query: str,
    ano: Optional[str] = None,
    nivel: Optional[str] = None,
    top_k: int = 10,
) -> str:
    """
    Busca legislação ANEEL via busca híbrida (dense+sparse RRF) com reranker.

    Args:
        query:  Pergunta ou termos de busca em português.
        ano:    Filtro por ano. Valores: "2016", "2021", "2022".
                Múltiplos: "2021,2022".
        nivel:  Nível do chunk: "L0" (ementa), "L1" (artigo/seção), "L2" (parágrafo).
        top_k:  Número de chunks a retornar (padrão 10, máximo 20).

    Returns:
        String formatada com os chunks mais relevantes e suas fontes.
    """
    top_k = min(int(top_k), 20)
    filt = _build_filter(ano, nivel)
    pontos = _hybrid_search(query, filt, top_rrf=50, top_k=top_k)
    if not pontos:
        return "Nenhum documento encontrado para a consulta."
    return _format_chunks(pontos, top_k)


# ── Ferramenta 2 ──────────────────────────────────────────────────────────────

def buscar_por_artigo(
    numero_artigo: str,
    ano: Optional[str] = None,
    tipo_documento: Optional[str] = None,
) -> str:
    """
    Busca chunks que referenciam um artigo específico da legislação ANEEL.

    Args:
        numero_artigo:    Número do artigo (ex: "45", "art. 45", "artigo 45").
        ano:              Filtro por ano: "2016", "2021" ou "2022".
        tipo_documento:   Tipo do ato normativo (ex: "resolução", "despacho", "portaria").

    Returns:
        Chunks que mencionam o artigo solicitado com fonte e contexto.
    """
    num = numero_artigo.strip().lower()
    num = num.replace("artigo", "").replace("art.", "").replace("art", "").strip()
    query = f"artigo {num} {tipo_documento or ''} {ano or ''}".strip()

    filt = _build_filter(ano, nivel=None)
    pontos = _hybrid_search(query, filt, top_rrf=50, top_k=10)
    if not pontos:
        return f"Nenhum documento encontrado para o artigo {numero_artigo}."
    return _format_chunks(pontos)


# ── Ferramenta 3 ──────────────────────────────────────────────────────────────

def resumir_documento(doc_id: str) -> str:
    """
    Recupera todos os chunks de um documento e retorna seu conteúdo completo
    para que o agente possa elaborar um resumo.

    Args:
        doc_id: Identificador do documento (ex: "2021__adsp20211699_1").

    Returns:
        Conteúdo concatenado de todos os chunks do documento, ordenado por nível.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = _get_qdrant()
    filt = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=filt,
        limit=200,
        with_payload=True,
    )
    if not results:
        return f"Documento '{doc_id}' não encontrado na coleção."

    nivel_ordem = {"L0": 0, "L1": 1, "L2": 2}
    results.sort(key=lambda p: nivel_ordem.get(p.payload.get("nivel", "L2"), 2))

    payload0 = results[0].payload or {}
    cabecalho = (
        f"Documento: {doc_id}\n"
        f"Ano: {payload0.get('ano', '?')} | "
        f"Tipo: {payload0.get('tipo_documento', '?')} | "
        f"Total de chunks: {len(results)}\n"
        f"{'=' * 60}\n\n"
    )
    corpo = "\n\n".join(
        f"[{p.payload.get('nivel', '?')}] {p.payload.get('text', '')}"
        for p in results
    )
    return cabecalho + corpo


# ── Ferramenta 4 ──────────────────────────────────────────────────────────────

def listar_tipos_documentos(ano: Optional[str] = None) -> str:
    """
    Lista os tipos de documentos disponíveis no corpus ANEEL RAG,
    com contagem por tipo e por ano.

    Args:
        ano: Filtro opcional por ano: "2016", "2021" ou "2022".

    Returns:
        Resumo dos tipos de documentos disponíveis e suas quantidades.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = _get_qdrant()

    filt = None
    if ano and ano.strip() in ANOS_VALIDOS:
        filt = Filter(
            must=[FieldCondition(key="ano", match=MatchValue(value=ano.strip()))]
        )

    contagem: dict = {}
    offset = None
    amostras = 0
    while amostras < 5000:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=filt,
            limit=250,
            offset=offset,
            with_payload=["tipo_documento", "ano"],
        )
        if not batch:
            break
        for p in batch:
            tipo = p.payload.get("tipo_documento", "desconhecido")
            ano_doc = p.payload.get("ano", "?")
            chave = f"{tipo} ({ano_doc})"
            contagem[chave] = contagem.get(chave, 0) + 1
        amostras += len(batch)
        if offset is None:
            break

    if not contagem:
        return "Nenhum documento encontrado."

    linhas = ["Tipos de documentos no corpus ANEEL RAG (amostra):\n"]
    for chave, qtd in sorted(contagem.items(), key=lambda x: -x[1]):
        linhas.append(f"  {chave}: {qtd} chunks")
    linhas.append(f"\nTotal amostrado: {amostras} chunks de 281.003")
    return "\n".join(linhas)


# ── Ferramenta 5 ──────────────────────────────────────────────────────────────

def buscar_documentos_revogados(
    ano: Optional[str] = None,
    tipo_documento: Optional[str] = None,
) -> str:
    """
    Busca documentos revogados/inativos no corpus ANEEL RAG.

    Args:
        ano:             Filtro por ano: "2016", "2021" ou "2022".
        tipo_documento:  Tipo do ato (ex: "resolução", "portaria").

    Returns:
        Lista de documentos revogados com metadados.
    """
    client = _get_qdrant()
    filt = _build_filter(
        ano=ano,
        nivel="metadata",
        is_active=False,
        tipo_documento=tipo_documento,
    )
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=filt,
        limit=10,
        with_payload=True,
    )
    if not results:
        return "Nenhum documento revogado encontrado."
    linhas = [f"Documentos revogados em {ano or 'todos os anos'} ({len(results)} encontrados):\n"]
    for p in results:
        payload = p.payload or {}
        conteudo = payload.get("conteudo", "")
        if conteudo:
            linhas.append(f"- {conteudo}")
        else:
            linhas.append(
                f"- {payload.get('doc_id', '?')} | "
                f"Situação: {payload.get('situacao', '?')} | "
                f"Publicação: {payload.get('publicacao', '?')}"
            )
    return "\n".join(linhas)


# ── Ferramenta 6 ──────────────────────────────────────────────────────────────

def buscar_por_metadados(
    autor: Optional[str] = None,
    assunto: Optional[str] = None,
    tipo_documento: Optional[str] = None,
    ano: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> str:
    """
    Busca documentos no corpus ANEEL por metadados estruturados.

    Args:
        autor:           Nome do autor ou órgão emissor.
        assunto:         Assunto ou tema do documento.
        tipo_documento:  Tipo do ato (ex: "resolução normativa", "portaria").
        ano:             Filtro por ano: "2016", "2021" ou "2022".
        is_active:       True = vigentes, False = revogados/inativos.

    Returns:
        Lista de documentos correspondentes com metadados.
    """
    client = _get_qdrant()
    filt = _build_filter(
        ano=ano,
        nivel="metadata",
        is_active=is_active,
        autor=autor,
        assunto=assunto,
        tipo_documento=tipo_documento,
    )
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=filt,
        limit=10,
        with_payload=True,
    )
    if not results:
        return "Nenhum documento encontrado com os filtros informados."

    linhas = [f"{len(results)} documento(s) encontrado(s):\n"]
    for p in results:
        payload = p.payload or {}
        linhas.append(
            f"- {payload.get('doc_id', '?')} | "
            f"Ano: {payload.get('ano', '?')} | "
            f"Tipo: {payload.get('tipo_documento', '?')} | "
            f"Autor: {payload.get('autor', '?')} | "
            f"Ativo: {payload.get('is_active', '?')} | "
            f"Assunto: {payload.get('assunto', '?')}"
        )
    return "\n".join(linhas)
