# Desafio ANEEL-RAG - Inteligência Artificial (UFG)

Repositório para o desenvolvimento de um sistema RAG (Retrieval-Augmented Generation) aplicado a normativas da ANEEL, utilizando o modelo GAIA 4B.

## 👥 Equipe
* **Gabriel Vidal**
* **Pablo Henrique**
* **Cissa Fernandes**

## 📂 Organização do Repositório
O projeto está estruturado para refletir o pipeline de dados:

* **`notebooks/`**: Contém todo o histórico de desenvolvimento no Google Colab, dividido por etapas:
    * `etapa1_ingestao`: Scripts de coleta.
    * `etapa2_processamento`: Tratamento de dados por anos (2016, 2021, 2022).
    * `etapa3_chunking`: Estratégias de quebra de texto.
    * `etapa4_embedding_e_busca`: Geração de vetores e implementação de Reranker.
* **`logs/`**: Registros de execução e métricas de performance.
* **`src/`**: Scripts Python finais para o deploy na VM.

## 🛠️ Tecnologias e Infra
* **LLM:** GAIA 4B (via vLLM)
* **Embeddings:** Modelos otimizados para o setor elétrico.
* **Infra:** Google Cloud Platform (VM com GPU L4).
* **SO:** Ubuntu 24.04 LTS.