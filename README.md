# ANEEL RAG — Desafio Agentes NLP 2025

Sistema de Recuperação e Geração Aumentada (RAG) sobre o corpus normativo da ANEEL (2016, 2021 e 2022), com busca híbrida densa+esparsa, reranker e geração via modelo GAIA 4B rodando localmente em GPU.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│  PIPELINE DE INGESTÃO (offline · Google Colab)                  │
│                                                                 │
│  Etapa 1 · Ingestão       26.739 PDFs da ANEEL                 │
│           Cloudflare bloqueou Python → solução: Chrome          │
│           DevTools fetch() · 35 arquivos irrecuperáveis (404)   │
│                    ↓                                            │
│  Etapa 2 · Extração       Docling · gate ≥ 0.80                 │
│           conteúdo(55%) + estrutura(15%) + cobertura(30%)       │
│           2016: 99.4% · 2021: 99.9% · 2022: 100.0%             │
│           score médio 0.958 · fallback: Qwen2.5-VL-7B          │
│                    ↓                                            │
│  Etapa 3 · Chunking       hierárquico L0/L1/L2                  │
│           L0: ementa · L1: artigo/seção · L2: parágrafo         │
│           281.003 chunks totais                                 │
│                    ↓                                            │
│  Etapa 4 · Embedding      BAAI/bge-m3 (dense 1024d + sparse)   │
│           upsert no Qdrant · índices por ano e nível            │
└─────────────────────────────────────────────────────────────────┘
                    ↓ dados persistidos
┌─────────────────────────────────────────────────────────────────┐
│  INFRAESTRUTURA DE SERVING (GCP · VM g2-standard-8 · L4 24GB)  │
│                                                                 │
│  Docker Compose · 4 serviços                                    │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   qdrant     │  │    gaia      │  │      embedding       │  │
│  │  v1.9.2      │  │  vLLM 0.19+  │  │  FastAPI + BGE-M3    │  │
│  │  :6333       │  │  :8080       │  │  + reranker  :8082   │  │
│  │  281k chunks │  │  GAIA-4B     │  │  25% VRAM (~6GB)     │  │
│  │  dense+sparse│  │  bfloat16    │  │                      │  │
│  │              │  │  70% VRAM    │  │                      │  │
│  │              │  │  (~16.8GB)   │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
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

### Alocação de VRAM (NVIDIA L4 · 24GB)

| Serviço | Mecanismo | VRAM |
|---|---|---|
| GAIA-4B (vLLM) | `GPU_MEMORY_UTILIZATION=0.70` | ~16.8GB |
| BGE-M3 + reranker | `torch.cuda.set_per_process_memory_fraction(0.25)` | ~6.0GB |
| Livre | — | ~1.2GB |

---

## Estrutura do Repositório

```
aneel-rag/
├── setup_vm.sh              # setup completo da VM (Docker + clone + compose)
├── docker-compose.yml       # 4 serviços: qdrant · gaia · embedding · adk
├── Dockerfile.gaia          # vLLM + GAIA-4B (CUDA 12.1 devel)
├── Dockerfile.embedding     # FastAPI + BGE-M3 + reranker (PyTorch 2.6)
├── Dockerfile.adk           # Google ADK (sem GPU)
├── .env.example             # variáveis de ambiente (copie para .env)
├── scripts/
│   ├── serve_gaia.py        # servidor vLLM com auto-download do modelo
│   ├── embedding_server.py  # FastAPI: /embed e /rerank
│   ├── agent.py             # agente ADK com 4 ferramentas RAG
│   ├── rag_tools.py         # busca híbrida RRF + reranker via HTTP
│   └── migrate_qdrant.py    # migração Qdrant Cloud → local (scroll)
├── notebooks/
│   ├── etapa1_ingestao/     # download dos PDFs via Chrome DevTools
│   ├── etapa2_processamento/# extração com Docling + gate de qualidade
│   ├── etapa3_chunking/     # chunking hierárquico L0/L1/L2
│   └── etapa4_embedding/    # embedding BGE-M3 + upsert Qdrant
└── logs/                    # logs de ingestão e embedding
```

---

## Pré-requisitos

- VM GCP com NVIDIA L4 (24GB VRAM), Ubuntu 22.04 (Deep Learning VM)
- Recomendado: `g2-standard-8` (8 vCPUs, 32GB RAM, 250GB SSD)
- Acesso ao Google Cloud Console
- Git instalado localmente

---

## Como Rodar

### 1. Subir a VM no GCP

Ligue a VM pelo [Google Cloud Console](https://console.cloud.google.com/compute/instances) e conecte via SSH no navegador.

### 2. Setup completo (uma só vez)

```bash
curl -fsSL https://raw.githubusercontent.com/vidal56/aneel-rag/main/setup_vm.sh | bash
```

O script instala Docker, NVIDIA Container Toolkit, clona o repositório e sobe os containers. Quando pausar para editar o `.env`, preencha:

```bash
# Apenas para migração inicial do Qdrant Cloud (opcional)
QDRANT_CLOUD_URL=https://SEU_CLUSTER.cloud.qdrant.io
QDRANT_CLOUD_APIKEY=SUA_CHAVE
```

### 3. Aguardar os containers subirem

```bash
cd ~/aneel-rag
docker compose logs -f gaia      # aguarda: "Application startup complete"
docker compose logs -f embedding # aguarda: "Serviço de embedding pronto"
docker compose ps                # todos devem estar "Up"
```

O GAIA-4B (~8GB) é baixado automaticamente na primeira execução — pode levar 10-20 min.

### 4. Popular o Qdrant (se banco vazio)

**Opção A — Reindexar via notebooks (recomendado):**

Abra os notebooks na pasta `notebooks/etapa4_embedding/` no Google Colab com GPU T4, apontando para o IP da VM:

```python
QDRANT_URL    = 'http://IP_DA_VM:6333'
QDRANT_APIKEY = ''
```

Rode na ordem: `2016` → `2021` → `2022`.

> Antes de rodar, abra a porta 6333 no firewall do GCP:
> VPC Network → Firewall → Criar regra → TCP 6333

**Opção B — Migrar do Qdrant Cloud (requer plano pago para snapshots):**

```bash
docker compose exec adk python3 migrate_qdrant.py
```

### 5. Verificar e acessar

```bash
# Checar quantidade de pontos no banco
curl http://localhost:6333/collections/aneel_rag | python3 -m json.tool | grep points_count

# Testar GAIA
curl http://localhost:8080/health

# Testar embedding
curl http://localhost:8082/health
```

Acesse a interface do agente no browser:
```
http://IP_DA_VM:8081
```

> Libere a porta 8081 no firewall do GCP se quiser acesso externo:
> VPC Network → Firewall → Criar regra → TCP 8081

---

## Ferramentas do Agente

O agente ADK expõe 4 ferramentas RAG:

| Ferramenta | Descrição |
|---|---|
| `buscar_legislacao_aneel` | Busca híbrida (dense+sparse RRF) com reranker · filtros por ano e nível |
| `buscar_por_artigo` | Busca chunks por número de artigo específico |
| `resumir_documento` | Recupera todos os chunks de um documento pelo doc_id |
| `listar_tipos_documentos` | Lista tipos de atos normativos disponíveis no corpus |

---

## Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| LLM | CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it via vLLM 0.19+ |
| Embedding | BAAI/bge-m3 (dense 1024d + sparse) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Banco vetorial | Qdrant 1.9.2 (Docker local) |
| Framework agente | Google ADK 0.5.0 |
| Extração PDF | Docling |
| Infraestrutura | Docker Compose · GCP · NVIDIA L4 |

---

## Corpus

| Ano | Documentos | Chunks |
|---|---|---|
| 2016 | 5.962 | ~90k |
| 2021 | 9.514 | ~100k |
| 2022 | 7.821 | ~91k |
| **Total** | **23.297** | **281.003** |

Nível de chunk: **L0** (ementa) · **L1** (artigo/seção) · **L2** (parágrafo)

---

## Segurança — Fechar porta 6333 após indexação

Durante a indexação via Colab (Etapa 4), a porta 6333 do Qdrant precisa estar aberta no firewall do GCP. **Após concluir o upsert, feche a regra imediatamente** para evitar que o banco fique exposto publicamente.

> O Qdrant local não tem autenticação configurada (`QDRANT_APIKEY` vazio). A porta só deve ficar aberta pelo tempo estritamente necessário para a indexação.

**Opção A — Pelo console do GCP (sem instalar nada):**

1. Acesse [VPC Network → Firewall](https://console.cloud.google.com/networking/firewalls/list) no Google Cloud Console
2. Localize a regra que libera a porta 6333 (ex: `allow-qdrant-6333`)
3. Marque a caixa ao lado da regra e clique em **Excluir**

**Opção B — Pelo terminal (via SSH na VM ou gcloud local):**

```bash
# Verificar regras de firewall existentes
gcloud compute firewall-rules list --filter="name~qdrant"

# Deletar a regra que abre a porta 6333
gcloud compute firewall-rules delete allow-qdrant-6333

# Confirmar que o Qdrant ainda funciona internamente (sem expor ao exterior)
curl http://localhost:6333/healthz
```

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

# Reiniciar um serviço
docker compose restart gaia

# Desligar VM (economizar)
gcloud compute instances stop gaia-aneel-server --zone=us-central1-a
```
