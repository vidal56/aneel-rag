#!/bin/bash
# =============================================================================
# ANEEL RAG — Setup completo da VM GCP (L4 · Ubuntu 22.04)
# Instala Docker, clona o repo e sobe todos os serviços via Docker Compose
# =============================================================================
# VM: g2-standard-8 · NVIDIA L4 24GB VRAM · 32GB RAM · 250GB SSD
#
# COMO USAR (uma só vez na VM):
#   curl -fsSL https://raw.githubusercontent.com/vidal56/aneel-rag/main/setup_vm.sh | bash
#   -- ou --
#   git clone https://github.com/vidal56/aneel-rag.git && bash aneel-rag/setup_vm.sh
# =============================================================================

set -e

echo "============================================================"
echo " ANEEL RAG — Setup VM"
echo "============================================================"

# ── 1. Instalar Docker (oficial) ────────────────────────────────
echo "[1/6] Instalando Docker..."

sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update -qq
sudo apt-get install -y -qq \
    docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker "$USER"

# ── 2. Instalar utilitários ─────────────────────────────────────
echo "[2/6] Instalando git, neovim, nvtop..."
sudo apt-get install -y -qq git neovim nvtop htop

# ── 3. Instalar NVIDIA Container Toolkit ────────────────────────
echo "[3/6] Instalando NVIDIA Container Toolkit..."
# Necessário para expor a GPU L4 dentro dos containers Docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update -qq
sudo apt-get install -y -qq nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# ── 4. Clonar repositório ───────────────────────────────────────
echo "[4/6] Clonando repositório..."
REPO_DIR="$HOME/aneel-rag"

if [ -d "$REPO_DIR" ]; then
    echo "  Repo já existe em $REPO_DIR — atualizando com git pull..."
    git -C "$REPO_DIR" pull
else
    git clone https://github.com/vidal56/aneel-rag.git "$REPO_DIR"
fi

cd "$REPO_DIR"

# ── 5. Configurar .env ──────────────────────────────────────────
echo "[5/6] Configurando .env..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │  ATENÇÃO: preencha as credenciais antes de subir    │"
    echo "  │  os containers:                                      │"
    echo "  │                                                      │"
    echo "  │  nano $REPO_DIR/.env                                 │"
    echo "  │                                                      │"
    echo "  │  Campos obrigatórios:                                │"
    echo "  │    GOOGLE_API_KEY   — Google AI Studio               │"
    echo "  │                                                      │"
    echo "  │  Para migrar dados do Qdrant Cloud pro local:        │"
    echo "  │    preencha também QDRANT_CLOUD_URL e               │"
    echo "  │    QDRANT_CLOUD_APIKEY antes de rodar o migrate.     │"
    echo "  └─────────────────────────────────────────────────────┘"
    echo ""
    echo "  Abrindo .env para edição (Ctrl+X para sair do nano)..."
    sleep 2
    nano .env
else
    echo "  .env já existe — mantendo configuração atual."
fi

# ── 6. Subir containers ─────────────────────────────────────────
echo "[6/6] Construindo e subindo containers..."
echo "  Isso pode levar 10-20 min na primeira vez (download de modelos)."
echo ""

# Usa newgrp para garantir que o grupo docker está ativo sem logout
newgrp docker <<DOCKERCMD
cd $REPO_DIR
docker compose up --build -d
DOCKERCMD

echo ""
echo "============================================================"
echo " Setup concluído!"
echo "============================================================"
echo ""
echo " Serviços:"
echo "   Qdrant    → http://localhost:6333"
echo "   GAIA vLLM → http://localhost:8080/v1"
echo "   Embedding → http://localhost:8082"
echo "   ADK Web   → http://$(curl -s ifconfig.me 2>/dev/null || echo 'IP_DA_VM'):8081"
echo ""
echo " Comandos úteis:"
echo "   docker compose logs -f gaia       ← logs do modelo"
echo "   docker compose logs -f embedding  ← logs do embedding"
echo "   docker compose logs -f adk        ← logs do agente"
echo "   docker compose ps                 ← status dos serviços"
echo "   nvtop                             ← GPU em tempo real"
echo ""
echo " Migrar dados Qdrant Cloud → local (rodar UMA vez):"
echo "   docker compose exec adk python3 migrate_qdrant.py"
echo ""
echo " Desligar VM (economizar):"
echo "   gcloud compute instances stop aneel-gaia-server --zone=us-central1-b"
echo "============================================================"
