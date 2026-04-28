"""
agent.py — Interface ADK para ANEEL RAG
Google ADK 0.5.0 · BGE-M3 fp16 · Qdrant local · Python 3.12
Modelo: CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it via vLLM (OpenAI-compat)
        Consumido via LiteLLM apontando para http://gaia:8080/v1
Uso:
    adk web            → UI web em http://0.0.0.0:8081
    adk run agent.py   → modo terminal interativo
"""
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from .rag_tools import (
    buscar_legislacao_aneel,
    buscar_por_artigo,
    resumir_documento,
    listar_tipos_documentos,
    buscar_documentos_revogados,
    buscar_por_metadados,
)
load_dotenv()

GAIA_API_URL = os.getenv("GAIA_API_URL", "http://gaia:8080/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gaia-4b")

_gaia = LiteLlm(
    model=f"openai/{MODEL_NAME}",
    api_base=GAIA_API_URL,
    api_key="not-needed",
)
_sabia = LiteLlm(
    model="openai/sabia-3",
    api_base="https://chat.maritaca.ai/api/v1",  # Adicionado o /v1 aqui
    api_key=os.getenv("MARITACA_API_KEY"),
)

root_agent = Agent(
    name="aneel_rag",
    model=_sabia,
    description=(
        "Agente RAG para legislação ANEEL usando GAIA 4B local. "
        "Busca documentos no Qdrant e responde em português."
    ),
    instruction=(
        "IMPORTANTE: Responda DIRETAMENTE sem explicar o que você vai fazer, "
        "quais funções vai usar ou seu processo de pensamento. Vá direto à resposta.\n\n"
        "Você é um assistente especializado em legislação e regulação do setor elétrico "
        "brasileiro, com foco nos atos normativos da ANEEL (2016, 2021 e 2022).\n\n"
        "Ao receber uma pergunta:\n"
        "1. Use buscar_legislacao_aneel para recuperar os chunks mais relevantes.\n"
        "   - Se o usuário filtrar por ano (2016, 2021 ou 2022), passe o parâmetro ano=.\n"
        "   - Se pedir nível específico (ementa, artigo, parágrafo), passe nivel= (L0/L1/L2).\n"
        "2. Se o usuário pedir busca por artigo específico, use buscar_por_artigo.\n"
        "3. Se o usuário pedir resumo de um documento pelo doc_id, use resumir_documento.\n"
        "4. Se o usuário quiser saber os tipos de atos disponíveis, use listar_tipos_documentos.\n"
        "5. Se o usuário perguntar sobre documentos revogados ou inativos, use buscar_documentos_revogados.\n"
        "6. Se o usuário buscar por autor, assunto ou tipo de documento, use buscar_por_metadados.\n"
        "7. Responda SOMENTE com base nos documentos recuperados como contexto.\n"
        "8. Sempre inclua as fontes (doc_id + ano) ao final da resposta.\n"
        "9. Ao listar documentos, formate cada item em uma linha separada assim:\n"
        "   - [doc_id] | Ano: [ano] | Tipo: [tipo] | Situação: [situacao]\n"
        "10. Nunca abrevie, resuma ou altere os doc_ids retornados pelas ferramentas.\n"
        "Responda em português do Brasil."
    ),

    tools=[
        buscar_legislacao_aneel,
        buscar_por_artigo,
        resumir_documento,
        listar_tipos_documentos,
        buscar_documentos_revogados,
        buscar_por_metadados,
    ],
)
