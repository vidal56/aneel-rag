"""
test_api.py — Testa o servidor GAIA após subir

Uso:
    python3 test_api.py                        # testa local
    python3 test_api.py --host 34.12.34.56     # testa VM remota
"""
import argparse
import json
import time
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", default="8080")
args = parser.parse_args()

BASE = f"http://{args.host}:{args.port}/v1"

print(f"Testando API em: {BASE}")
print("=" * 60)

# ── 1. Health check ────────────────────────────────────────────
print("\n[1/3] Health check...")
try:
    r = requests.get(f"http://{args.host}:{args.port}/health", timeout=10)
    print(f"  Status: {r.status_code} {r.text}")
except Exception as e:
    print(f"  FALHOU: {e}")
    print("  O servidor está rodando? Execute: python3 serve_gaia.py")
    exit(1)

# ── 2. Listar modelos ──────────────────────────────────────────
print("\n[2/3] Modelos disponíveis...")
r = requests.get(f"{BASE}/models")
modelos = r.json().get("data", [])
for m in modelos:
    print(f"  - {m['id']}")

# ── 3. Geração de texto com contexto ANEEL ─────────────────────
print("\n[3/3] Teste de geração com contexto jurídico ANEEL...")

contexto = """[DOCUMENTO 1]
Fonte: 2021__adsp20211699_1 | Ano: 2021 | Nível: L2 | Score: 0.9969
Nos termos do art. 45 c/c o art. 48 da Resolução Normativa nº 273, de 2007, que disciplina
o processo administrativo na ANEEL, é cabível recurso administrativo contra atos dos
Superintendentes com delegação de poder decisório no âmbito da ANEEL, sendo de 10 dias
o prazo para interposição de recurso, contado a partir da cientificação oficial."""

payload = {
    "model": "gaia-4b",
    "messages": [
        {
            "role": "system",
            "content": (
                "Você é um assistente especializado em legislação da ANEEL. "
                "Responda SOMENTE com base nos documentos fornecidos. "
                "Cite sempre o doc_id e o ano da fonte. "
                "Responda em português do Brasil."
            )
        },
        {
            "role": "user",
            "content": (
                f"Use o contexto abaixo para responder:\n\n"
                f"CONTEXTO:\n{contexto}\n\n"
                f"PERGUNTA: Qual o prazo para recurso administrativo na ANEEL?"
            )
        }
    ],
    "max_tokens": 256,
    "temperature": 0.1,
    "stream": False,
}

t0 = time.time()
r = requests.post(f"{BASE}/chat/completions", json=payload, timeout=60)
t_total = time.time() - t0

if r.status_code == 200:
    data = r.json()
    resposta = data["choices"][0]["message"]["content"]
    tokens_in  = data["usage"]["prompt_tokens"]
    tokens_out = data["usage"]["completion_tokens"]
    tok_s = tokens_out / t_total

    print(f"\n  Resposta:\n  {resposta}")
    print(f"\n  Tokens entrada : {tokens_in}")
    print(f"  Tokens saída   : {tokens_out}")
    print(f"  Tempo total    : {t_total:.1f}s")
    print(f"  Velocidade     : {tok_s:.1f} tokens/s")
    print("\n  API funcionando corretamente!")
else:
    print(f"  ERRO {r.status_code}: {r.text}")
