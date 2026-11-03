"""Baixa o B2W-Reviews01 e gera treino, validação e teste no formato de chat do mlx-lm.

    uv run python -m avaliacoes.preparar

A divisão é por produto: todas as avaliações de um produto caem na mesma
parte. Dividir por avaliação deixaria o modelo ver no treino outras opiniões
sobre o mesmo item do teste, e a nota medida seria otimista.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from avaliacoes import prompt

RAIZ = Path(__file__).resolve().parent.parent
BRUTO = RAIZ / "dados" / "brutos" / "b2w-reviews01.parquet"
SAIDA = RAIZ / "dados"

# Conversão para Parquet que o Hugging Face publica do dataset, fixada num
# commit para que o arquivo não mude por baixo dos números.
URL = (
    "https://huggingface.co/datasets/ruanchaves/b2w-reviews01/resolve/"
    "c95a67fea418ab885cc673055cb46d4b7bdd47fa/default/train/0000.parquet"
)
SHA256 = "4fdba9269b3a75efc9a59beb0ac00914e7205d6d1190a56a0bdfbacfbc3b6947"

SEMENTE = 20261103
# Parte de cada produto, pelo hash do id: 0–79 treino, 80–89 validação, 90–99 teste.
FAIXAS = {"treino": range(0, 80), "validacao": range(80, 90), "teste": range(90, 100)}
AMOSTRAS = {"treino": 4000, "validacao": 400, "teste": 1000}
# O p99 do texto é 643 caracteres; acima disso o corte só tira cauda.
MAXIMO_DE_CARACTERES = 1000


def baixar() -> None:
    if not BRUTO.exists():
        BRUTO.parent.mkdir(parents=True, exist_ok=True)
        print(f"baixando {URL}")
        urllib.request.urlretrieve(URL, BRUTO)
    digest = hashlib.sha256(BRUTO.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"SHA-256 inesperado para {BRUTO}: {digest}")


def parte_do_produto(product_id: str) -> str:
    balde = int(hashlib.sha256(product_id.encode()).hexdigest(), 16) % 100
    return next(nome for nome, faixa in FAIXAS.items() if balde in faixa)


def chave_de_duplicata(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip().lower()


def limpar(linhas: list[dict]) -> tuple[list[dict], Counter]:
    """Descarta o que não tem rótulo ou texto e repetições do mesmo texto.

    A duplicata sai antes da divisão: o mesmo "Produto excelente, recomendo"
    em dois produtos poderia cair um no treino e outro no teste.
    """
    descartes = Counter()
    vistos = set()
    limpas = []
    for linha in linhas:
        texto = (linha["review_text"] or "").strip()
        if not texto:
            descartes["sem texto"] += 1
            continue
        if linha["recommend_to_a_friend"] not in ("Yes", "No"):
            descartes["sem recomendação"] += 1
            continue
        if linha["overall_rating"] not in range(1, 6):
            descartes["nota fora de 1 a 5"] += 1
            continue
        chave = chave_de_duplicata(texto)
        if chave in vistos:
            descartes["texto repetido"] += 1
            continue
        vistos.add(chave)
        limpas.append({
            "produto": linha["product_id"],
            "titulo": (linha["review_title"] or "").strip(),
            "texto": texto[:MAXIMO_DE_CARACTERES],
            "nota": linha["overall_rating"],
            "recomenda": linha["recommend_to_a_friend"] == "Yes",
        })
    return limpas, descartes


def exemplo(linha: dict) -> dict:
    return {
        "messages": prompt.mensagens(linha["titulo"], linha["texto"])
        + [{"role": "assistant", "content": prompt.resposta(linha["nota"], linha["recomenda"])}]
    }


def main() -> None:
    baixar()
    linhas = pq.read_table(BRUTO).to_pylist()
    limpas, descartes = limpar(linhas)

    partes = {nome: [] for nome in FAIXAS}
    for linha in limpas:
        partes[parte_do_produto(linha["produto"])].append(linha)

    gerador = random.Random(SEMENTE)
    manifesto = {
        "fonte": URL,
        "sha256": SHA256,
        "semente": SEMENTE,
        "linhas_originais": len(linhas),
        "descartes": dict(descartes),
        "linhas_limpas": len(limpas),
        "partes": {},
    }
    nomes_de_arquivo = {"treino": "train", "validacao": "valid", "teste": "test"}
    for nome, conjunto in partes.items():
        # Amostra simples, sem balancear: o teste tem a distribuição real das
        # notas, e o treino também, para não ensinar ao modelo uma frequência
        # de notas que ele não vai encontrar.
        amostra = gerador.sample(conjunto, AMOSTRAS[nome])
        caminho = SAIDA / f"{nomes_de_arquivo[nome]}.jsonl"
        with caminho.open("w", encoding="utf-8") as arquivo:
            for linha in amostra:
                arquivo.write(json.dumps(exemplo(linha), ensure_ascii=False) + "\n")
        manifesto["partes"][nome] = {
            "arquivo": caminho.name,
            "pool": len(conjunto),
            "produtos_no_pool": len({l["produto"] for l in conjunto}),
            "amostra": len(amostra),
            "notas": dict(sorted(Counter(l["nota"] for l in amostra).items())),
            "recomenda": dict(Counter("sim" if l["recomenda"] else "não" for l in amostra)),
        }

    produtos = {nome: {l["produto"] for l in conjunto} for nome, conjunto in partes.items()}
    assert not (produtos["treino"] & produtos["teste"]), "produto em treino e teste"
    assert not (produtos["treino"] & produtos["validacao"]), "produto em treino e validação"

    (SAIDA / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifesto, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
