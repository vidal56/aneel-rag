# ANEEL RAG — Desafio Agentes NLP 2025

> Sistema de Recuperação e Geração Aumentada (RAG) sobre o corpus normativo da ANEEL (2016, 2021 e 2022), com busca híbrida densa+esparsa, reranker e geração via modelo GAIA 4B rodando localmente em GPU.

---

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Stack Tecnológica](#stack-tecnológica)
- [Corpus](#corpus)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Como Rodar](#como-rodar)
- [Ferramentas do Agente](#ferramentas-do-agente)
- [Pipeline de Ingestão](#pipeline-de-ingestão)
- [Alocação de VRAM](#alocação-de-vram)
- [Comandos Úteis](#comandos-úteis)
- [Troubleshooting](#troubleshooting)
- [Contribuindo](#contribuindo)

---

## Visão Geral

Este projeto implementa um agente conversacional especializado em legislação e regulação do setor elétrico brasileiro, com foco nos atos normativos da ANEEL dos anos de **2016, 2021 e 2022**.

O sistema combina:
- **Busca híbrida** (dense + sparse via RRF) com reranker neural para máxima relevância
- **281.003 chunks** hierárquicos indexados no Qdrant
- **Modelo GAIA 4B** (fine-tuned em português brasileiro) rodando localmente via vLLM
- **Google ADK 0.5.0** como framework do agente com 4 ferramentas RAG especializadas

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│  PIPELINE DE INGESTÃO (offline · Google Colab)                  │
│                                                                 │
│  Etapa 1 · Ingestão       26.739 PDFs da ANEEL                  │
│           Cloudflare bloqueou Python → solução: Chrome          │
│           DevTools fetch() · 35 arquivos irrecuperáveis (404)   │
│                    ↓                                            │
│  Etapa 2 · Extração       Docling · gate ≥ 0.80                 │
│           conteúdo(55%) + estrutura(15%) + cobertura(30%)       │
│           2016: 99.4% · 2021: 99.9% · 2022: 100.0%              │
│           score médio 0.958 · fallback: Qwen2.5-VL-7B           │
│                    ↓                                            │
│  Etapa 3 · Chunking       hierárquico L0/L1/L2                  │
│           L0: ementa · L1: artigo/seção · L2: parágrafo         │
│           281.003 chunks totais                                 │
│                    ↓                                            │
│  Etapa 4 · Embedding      BAAI/bge-m3 (dense 1024d + sparse)    │
│           upsert no Qdrant · índices por ano e nível            │
└─────────────────────────────────────────────────────────────────┘
                    ↓ dados persistidos
┌─────────────────────────────────────────────────────────────────┐
│  INFRAESTRUTURA DE SERVING (GCP · VM g2-standard-8 · L4 24GB)   │
│                                                                 │
│  Docker Compose · 4 serviços                                    │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   qdrant     │  │    gaia      │  │      embedding       │   │
│  │  v1.9.2      │  │  vLLM 0.19+  │  │  FastAPI + BGE-M3    │   │
│  │  :6333       │  │  :8080       │  │  + reranker  :8082   │   │
│  │  281k chunks │  │  GAIA-4B     │  │  25% VRAM (~6GB)     │   │
│  │  dense+sparse│  │  bfloat16    │  │                      │   │
│  │              │  │  70% VRAM    │  │                      │   │
│  │              │  │  (~16.8GB)   │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                          ↓ HTTP interno                         │
│                   ┌──────────────┐                              │
│                   │     adk      │                              │
│                   │ Google ADK   │                              │
│                   │  :8081       │                              │
│                   │  4 ferramentas RAG                          │
│                   └──────────────┘                              │
│                          ↓ HTTP público                         │
│                   http://IP_DA_VM:8081                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stack Tecnológica

| Componente | Tecnologia | Detalhes |
|---|---|---|
| LLM | CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it | via vLLM com API OpenAI-compat |
| Embedding | BAAI/bge-m3 | dense 1024d + sparse |
| Reranker | BAAI/bge-reranker-v2-m3 | reranking neural |
| Banco vetorial | Qdrant 1.9.2 | Docker local |
| Framework agente | Google ADK 0.5.0 | 4 ferramentas RAG |
| Extração PDF | Docling | gate de qualidade ≥ 0.80 |
| Infraestrutura | Docker Compose · GCP · NVIDIA L4 | VM g2-standard-8 |

---

## Corpus

| Ano | Documentos | Chunks |
|---|---|---|
| 2016 | 5.962 | ~90k |
| 2021 | 9.514 | ~100k |
| 2022 | 7.821 | ~91k |
| **Total** | **23.297** | **281.003** |

**Níveis de chunk:**
- **L0** — Ementa (resumo do documento)
- **L1** — Artigo ou seção
- **L2** — Parágrafo ou inciso

---

## Estrutura do Repositório

```
aneel-rag/
├── setup_vm.sh               # setup completo da VM (Docker + clone + compose)
├── docker-compose.yml        # 4 serviços: qdrant · gaia · embedding · adk
├── Dockerfile.gaia           # vLLM + GAIA-4B (CUDA 12.1 devel)
├── Dockerfile.embedding      # FastAPI + BGE-M3 + reranker (PyTorch 2.6)
├── Dockerfile.adk            # Google ADK (sem GPU)
├── .env.example              # variáveis de ambiente (copie para .env)
├── scripts/
│   ├── serve_gaia.py         # servidor vLLM com auto-download do modelo
│   ├── embedding_server.py   # FastAPI: /embed e /rerank
│   ├── agent.py              # agente ADK com 4 ferramentas RAG
│   ├── rag_tools.py          # busca híbrida RRF + reranker via HTTP
│   └── migrate_qdrant.py     # migração Qdrant Cloud → local (scroll)
├── notebooks/
│   ├── etapa1_ingestao/      # download dos PDFs via Chrome DevTools
│   ├── etapa2_processamento/ # extração com Docling + gate de qualidade
│   ├── etapa3_chunking/      # chunking hierárquico L0/L1/L2
│   └── etapa4_embedding/     # embedding BGE-M3 + upsert Qdrant
└── logs/                     # logs de ingestão e embedding
```

---

## Pré-requisitos

- VM GCP com **NVIDIA L4 (24GB VRAM)**, Ubuntu 22.04 (Deep Learning VM)
- Tipo recomendado: `g2-standard-8` (8 vCPUs, 32GB RAM, 250GB SSD)
- Acesso ao Google Cloud Console
- Git instalado

> ⚠️ A GPU L4 é obrigatória para rodar o GAIA-4B e o embedding ao mesmo tempo dentro do limite de VRAM.

---

## Como Rodar

### 1. Subir a VM no GCP

Ligue a VM pelo [Google Cloud Console](https://console.cloud.google.com/compute/instances) e conecte via SSH no navegador.

---

### 2. Setup completo (apenas na primeira vez)

```bash
curl -fsSL https://raw.githubusercontent.com/vidal56/aneel-rag/main/setup_vm.sh | bash
```

O script instala Docker, NVIDIA Container Toolkit, clona o repositório e sobe os containers automaticamente.

Quando pausar para editar o `.env`, preencha se necessário:

```bash
# Apenas para migração inicial do Qdrant Cloud (opcional)
QDRANT_CLOUD_URL=https://SEU_CLUSTER.cloud.qdrant.io
QDRANT_CLOUD_APIKEY=SUA_CHAVE
```

---

### 3. Aguardar os containers subirem

```bash
cd ~/aneel-rag
docker compose logs -f gaia       # aguarda: "Application startup complete"
docker compose logs -f embedding  # aguarda: "Serviço de embedding pronto"
docker compose ps                 # todos devem estar "Up"
```

> O GAIA-4B (~8GB) é baixado automaticamente na primeira execução — pode levar **10 a 20 minutos**.

---

### 4. Popular o Qdrant (se banco estiver vazio)

**Opção A — Reindexar via notebooks (recomendado):**

Abra os notebooks na pasta `notebooks/etapa4_embedding/` no Google Colab com GPU T4, apontando para o IP da VM:

```python
QDRANT_URL    = 'http://IP_DA_VM:6333'
QDRANT_APIKEY = ''
```

Rode na ordem: `2016` → `2021` → `2022`.

> Antes de rodar, abra a porta 6333 no firewall do GCP:
> VPC Network → Firewall → Criar regra → TCP 6333

**Opção B — Migrar do Qdrant Cloud:**

```bash
docker compose exec adk python3 migrate_qdrant.py
```

---

### 5. Verificar os serviços

```bash
# Quantidade de pontos no banco
curl http://localhost:6333/collections/aneel_rag | python3 -m json.tool | grep points_count

# Saúde do GAIA
curl http://localhost:8080/health

# Saúde do embedding
curl http://localhost:8082/health
```

---

### 6. Acessar o agente

Libere a porta 8081 no firewall do GCP se quiser acesso externo:

> GCP Console → VPC Network → Firewall → Criar regra → TCP 8081

Acesse no navegador:

```
http://IP_DA_VM:8081
```

---

## Ferramentas do Agente

O agente ADK expõe 4 ferramentas RAG especializadas:

| Ferramenta | Descrição |
|---|---|
| `buscar_legislacao_aneel` | Busca híbrida (dense+sparse RRF) com reranker · filtros por ano e nível (L0/L1/L2) |
| `buscar_por_artigo` | Busca chunks por número de artigo específico dentro do corpus |
| `resumir_documento` | Recupera todos os chunks de um documento a partir do `doc_id` |
| `listar_tipos_documentos` | Lista todos os tipos de atos normativos disponíveis no corpus |

**Exemplos de uso no chat:**

```
"Quais são os regulamentos para hifrelétricas?"
"Me mostre o artigo 5 da resolução normativa de 2022."
"Resuma o documento ANEEL-RN-1896-2022."
"Quais tipos de atos normativos existem no banco?"
```

---

## Pipeline de Ingestão

### Etapa 1 — Download dos PDFs

- **26.739 PDFs** baixados do portal da ANEEL
- Cloudflare bloqueou downloads via Python → solução via Chrome DevTools com `fetch()`
- **35 arquivos irrecuperáveis** (erro 404 no servidor da ANEEL)

### Etapa 2 — Extração de texto com Docling

- Gate de qualidade com score mínimo **≥ 0.80**
- Pesos: conteúdo (55%) + estrutura (15%) + cobertura (30%)
- Taxa de aprovação: **2016: 99.4% · 2021: 99.9% · 2022: 100.0%**
- Score médio geral: **0.958**
- Fallback para documentos com baixa qualidade: **Qwen2.5-VL-7B**

### Etapa 3 — Chunking Hierárquico

- **L0** → ementa (nível do documento)
- **L1** → artigo ou seção
- **L2** → parágrafo ou inciso
- Total: **281.003 chunks**

### Etapa 4 — Embedding e indexação

- Modelo: **BAAI/bge-m3** (dense 1024 dimensões + sparse)
- Indexação por ano e por nível no Qdrant
- Busca híbrida com **Reciprocal Rank Fusion (RRF)**
- Reranker: **BAAI/bge-reranker-v2-m3**

---

## Alocação de VRAM

GPU: **NVIDIA L4 · 24GB**

| Serviço | Mecanismo | VRAM |
|---|---|---|
| GAIA-4B (vLLM) | `GPU_MEMORY_UTILIZATION=0.70` | ~16.8GB |
| BGE-M3 + reranker | `torch.cuda.set_per_process_memory_fraction(0.25)` | ~6.0GB |
| Livre | — | ~1.2GB |

---

## Comandos Úteis

```bash
# Status dos containers
docker compose ps

# Logs em tempo real
docker compose logs -f gaia
docker compose logs -f embedding
docker compose logs -f adk

# Monitorar GPU
nvtop

# Reiniciar um serviço específico
docker compose restart gaia

# Parar todos os containers (economizar recursos)
docker compose down

# Subir todos os containers
docker compose up -d

# Desligar a VM pelo terminal (economizar créditos GCP)
gcloud compute instances stop gaia-aneel-server --zone=us-central1-a
```

---

## Troubleshooting

### `module 'aneel_rag' has no attribute 'agent'`

O `__init__.py` do pacote está vazio. Corrija com:

```bash
docker exec -it aneel_adk bash -c "echo 'from . import agent' > /app/aneel_rag/__init__.py"
docker restart aneel_adk
```

### `No module named 'rag_tools'`

O import no `agent.py` está absoluto. Corrija para relativo:

```bash
docker exec -it aneel_adk sed -i 's/from rag_tools import/from .rag_tools import/' /app/aneel_rag/agent.py
docker restart aneel_adk
```

### `"auto" tool choice requires --enable-auto-tool-choice`

O vLLM não está com suporte a tool calling habilitado. Adicione em `scripts/serve_gaia.py`:

```python
"--enable-auto-tool-choice",
"--tool-call-parser", "llama3_json",
```

Depois rebuilde o container:

```bash
docker compose up -d --build gaia
```

### Container não sobe após reinício da VM

```bash
cd ~/aneel-rag
docker compose up -d
docker compose logs -f  # acompanhe os logs
```

---

## Contribuindo

1. Faça um fork do repositório
2. Crie uma branch: `git checkout -b minha-feature`
3. Commit suas mudanças: `git commit -m "feat: descrição da mudança"`
4. Push para a branch: `git push origin minha-feature`
5. Abra um Pull Request

---

> Desenvolvido para o **Desafio Agentes NLP 2025** · Corpus: ANEEL 2016, 2021 e 2022