"""
migrate_qdrant.py — Migra a collection aneel_rag do Qdrant Cloud para o local

Fluxo:
  1. Cria snapshot da collection no Qdrant Cloud
  2. Faz download do snapshot (.snapshot)
  3. Faz upload para o Qdrant local (Docker)
  4. Restaura a collection localmente

Pré-requisito:
  - Qdrant local rodando: docker compose up qdrant -d
  - .env com QDRANT_CLOUD_URL e QDRANT_CLOUD_APIKEY preenchidos

Uso:
  docker compose exec adk python3 migrate_qdrant.py
  -- ou --
  python3 migrate_qdrant.py
"""

import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Configuração ──────────────────────────────────────────────────────────────

CLOUD_URL     = os.getenv("QDRANT_CLOUD_URL", "").rstrip("/")
CLOUD_APIKEY  = os.getenv("QDRANT_CLOUD_APIKEY", "")
LOCAL_URL     = os.getenv("QDRANT_LOCAL_URL", "http://qdrant:6333").rstrip("/")
COLLECTION    = os.getenv("QDRANT_COLLECTION", "aneel_rag")
SNAPSHOT_FILE = Path("/tmp/aneel_rag.snapshot")


def cloud_headers():
    return {"api-key": CLOUD_APIKEY, "Content-Type": "application/json"}


def check_config():
    if not CLOUD_URL:
        raise ValueError(
            "QDRANT_CLOUD_URL não configurado no .env\n"
            "Exemplo: QDRANT_CLOUD_URL=https://xxxxx.us-east4-0.gcp.cloud.qdrant.io"
        )
    if not CLOUD_APIKEY:
        raise ValueError("QDRANT_CLOUD_APIKEY não configurado no .env")


def step1_create_snapshot() -> str:
    """Cria snapshot no Qdrant Cloud e retorna o nome do arquivo."""
    print(f"[1/4] Criando snapshot da collection '{COLLECTION}' no Cloud...")
    url = f"{CLOUD_URL}/collections/{COLLECTION}/snapshots"
    r = requests.post(url, headers=cloud_headers(), timeout=300)
    r.raise_for_status()
    snapshot_name = r.json()["result"]["name"]
    print(f"  Snapshot criado: {snapshot_name}")
    return snapshot_name


def step2_download_snapshot(snapshot_name: str):
    """Faz download do snapshot para /tmp."""
    print(f"[2/4] Baixando snapshot (~pode demorar dependendo do tamanho)...")
    url = f"{CLOUD_URL}/collections/{COLLECTION}/snapshots/{snapshot_name}"
    r = requests.get(url, headers=cloud_headers(), stream=True, timeout=600)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(SNAPSHOT_FILE, "wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"  {pct:.1f}% ({downloaded // 1_000_000}MB / {total // 1_000_000}MB)", end="\r")

    print(f"\n  Download concluído: {SNAPSHOT_FILE} ({SNAPSHOT_FILE.stat().st_size // 1_000_000}MB)")


def step3_upload_to_local():
    """Faz upload do snapshot para o Qdrant local."""
    print(f"[3/4] Fazendo upload do snapshot para o Qdrant local...")
    url = f"{LOCAL_URL}/collections/{COLLECTION}/snapshots/upload?priority=snapshot"

    with open(SNAPSHOT_FILE, "rb") as f:
        r = requests.post(
            url,
            files={"snapshot": (SNAPSHOT_FILE.name, f, "application/octet-stream")},
            timeout=600,
        )
    r.raise_for_status()
    print(f"  Upload concluído. Resposta: {r.json()}")


def step4_verify():
    """Verifica se a collection foi restaurada corretamente."""
    print(f"[4/4] Verificando collection local...")
    # Aguarda indexação
    for attempt in range(10):
        r = requests.get(f"{LOCAL_URL}/collections/{COLLECTION}", timeout=30)
        if r.status_code == 200:
            info = r.json()["result"]
            points = info.get("points_count", 0)
            status = info.get("status", "?")
            print(f"  Collection: {COLLECTION}")
            print(f"  Pontos    : {points:,}")
            print(f"  Status    : {status}")
            if status == "green":
                return True
        print(f"  Aguardando indexação... (tentativa {attempt + 1}/10)")
        time.sleep(10)
    print("  AVISO: collection não ficou 'green' no tempo esperado.")
    print("  Verifique com: curl http://localhost:6333/collections/aneel_rag")
    return False


def main():
    print("=" * 60)
    print(" ANEEL RAG — Migração Qdrant Cloud → Local")
    print("=" * 60)
    print(f" Cloud  : {CLOUD_URL}")
    print(f" Local  : {LOCAL_URL}")
    print(f" Coleção: {COLLECTION}")
    print()

    check_config()

    snapshot_name = step1_create_snapshot()
    step2_download_snapshot(snapshot_name)
    step3_upload_to_local()
    ok = step4_verify()

    print()
    if ok:
        print("✓ Migração concluída com sucesso!")
        print(f"  Remova o snapshot temporário: rm {SNAPSHOT_FILE}")
        print()
        print("  Atualize o .env para apontar pro Qdrant local:")
        print("    QDRANT_URL=http://qdrant:6333")
        print("    QDRANT_APIKEY=")
    else:
        print("⚠ Migração finalizada mas verifique o status da collection.")


if __name__ == "__main__":
    main()
