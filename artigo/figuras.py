# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.0"]
# ///
"""Gera os gráficos do artigo, em português e em inglês, a partir de resultados/.

    uv run artigo/figuras.py

O Medium não mostra tabelas em Markdown; os números que importam vão em PNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = Path(__file__).resolve().parent / "figuras"

# Ênfase: o modelo ajustado na cor de destaque, os outros dois em cinza.
DESTAQUE = "#2a78d6"
CONTEXTO = "#a3a29b"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_APAGADA = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
FUNDO = "#fcfcfb"

TEXTOS = {
    "pt": {
        "modelos": {
            "base-0.8b": "Qwen3.5-0.8B\nsem ajuste",
            "lora-0.8b": "Qwen3.5-0.8B\n+ LoRA",
            "qwen3.5-9b-q4": "Qwen3.5-9B Q4\nsem ajuste",
        },
        "f1": "Nota: F1 macro",
        "rec": "Recomendação: acurácia",
        "sempre_5": "sempre 5",
        "sempre_rec": "sempre recomenda",
        "nota_rodape": "1.000 avaliações de teste do B2W-Reviews01 · barras: intervalo de 95% por bootstrap",
        "curva_titulo": "Perda por token da resposta",
        "curva_x": "atualizações do otimizador (lotes de 8)",
        "serie_val": "validação (400 avaliações)",
        "serie_treino": "treino",
        "decimal": ",",
    },
    "en": {
        "modelos": {
            "base-0.8b": "Qwen3.5-0.8B\nno tuning",
            "lora-0.8b": "Qwen3.5-0.8B\n+ LoRA",
            "qwen3.5-9b-q4": "Qwen3.5-9B Q4\nno tuning",
        },
        "f1": "Rating: macro F1",
        "rec": "Recommendation: accuracy",
        "sempre_5": "always 5",
        "sempre_rec": "always recommends",
        "nota_rodape": "1,000 B2W-Reviews01 test reviews · bars: 95% bootstrap interval",
        "curva_titulo": "Loss per answer token",
        "curva_x": "optimizer updates (batches of 8)",
        "serie_val": "validation (400 reviews)",
        "serie_treino": "training",
        "decimal": ".",
    },
}

ORDEM = ["base-0.8b", "lora-0.8b", "qwen3.5-9b-q4"]


def estilo() -> None:
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": EIXO,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.facecolor": FUNDO,
        "figure.facecolor": FUNDO,
        "xtick.color": TINTA_APAGADA,
        "ytick.color": TINTA_SECUNDARIA,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    })


def numero(valor: float, casas: int, decimal: str, porcento: bool = False) -> str:
    texto = f"{valor * 100 if porcento else valor:.{casas}f}".replace(".", decimal)
    return texto + ("%" if porcento else "")


def painel(eixo, relatorios, chave, titulo, referencia, rotulo_referencia, textos, porcento):
    for posicao, nome in enumerate(reversed(ORDEM)):
        metricas = relatorios[nome]["metricas"]
        if chave == "f1":
            valor = metricas["nota"]["f1_macro"]
            baixo, alto = metricas["intervalos_95"]["nota_f1_macro"]
        else:
            valor = metricas["recomenda"]["acuracia"]
            baixo, alto = metricas["intervalos_95"]["recomenda_acuracia"]
        cor = DESTAQUE if nome == "lora-0.8b" else CONTEXTO
        eixo.plot([baixo, alto], [posicao, posicao], color=cor, linewidth=2, solid_capstyle="round", zorder=2)
        eixo.scatter([valor], [posicao], s=70, color=cor, edgecolors=FUNDO, linewidths=2, zorder=3)
        eixo.annotate(
            numero(valor, 1 if porcento else 3, textos["decimal"], porcento),
            (alto, posicao), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=11, color=TINTA, fontweight="bold" if nome == "lora-0.8b" else "normal",
        )
    eixo.axvline(referencia, color=TINTA_APAGADA, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    eixo.annotate(
        rotulo_referencia, (referencia, len(ORDEM) - 0.45), xytext=(4, 0), textcoords="offset points",
        fontsize=9.5, color=TINTA_APAGADA, va="center",
    )
    eixo.set_yticks(range(len(ORDEM)), [textos["modelos"][n] for n in reversed(ORDEM)])
    eixo.tick_params(axis="y", length=0)
    eixo.set_title(titulo, loc="left", fontsize=12.5, color=TINTA, pad=12)
    eixo.grid(axis="x", color=GRADE, linewidth=0.8)
    eixo.set_axisbelow(True)
    eixo.set_ylim(-0.6, len(ORDEM) - 0.2)
    if porcento:
        eixo.xaxis.set_major_formatter(lambda x, _: numero(x, 0, textos["decimal"], True))
    else:
        eixo.xaxis.set_major_formatter(lambda x, _: numero(x, 1, textos["decimal"]))


def resultados(relatorios: dict, idioma: str) -> None:
    textos = TEXTOS[idioma]
    figura, (esquerda, direita) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    painel(esquerda, relatorios, "f1", textos["f1"], 0.115, textos["sempre_5"], textos, porcento=False)
    esquerda.set_xlim(0.0, 0.62)
    painel(direita, relatorios, "rec", textos["rec"], 0.764, textos["sempre_rec"], textos, porcento=True)
    direita.set_xlim(0.70, 1.0)
    figura.text(0.01, 0.01, textos["nota_rodape"], fontsize=9, color=TINTA_APAGADA)
    figura.tight_layout(rect=(0, 0.05, 1, 1), w_pad=3)
    figura.savefig(SAIDA / f"{idioma}-resultados.png", dpi=200)
    plt.close(figura)


def curva(idioma: str) -> None:
    textos = TEXTOS[idioma]
    pontos = [
        json.loads(linha) for linha in (RAIZ / "resultados" / "curva-de-treino.jsonl").read_text().splitlines()
    ]
    # O callback conta lotes de 4; a acumulação de 2 faz cada atualização valer 2 lotes.
    validacao = [((p["iteration"] + 1) // 2, p["val_loss"]) for p in pontos if p["tipo"] == "validacao"]
    # A perda de treino sai a cada 25 lotes, mais vezes que a de validação, e
    # mostra onde fica a queda entre 0 e 50 atualizações.
    treino = [(p["iteration"] / 2, p["train_loss"]) for p in pontos if p["tipo"] == "treino"]
    x, y = zip(*validacao)
    xt, yt = zip(*treino)

    figura, eixo = plt.subplots(figsize=(10, 3.4))
    eixo.plot(xt, yt, color=CONTEXTO, linewidth=1.5, zorder=1)
    eixo.plot(x, y, color=DESTAQUE, linewidth=2, zorder=2)
    eixo.scatter(x, y, s=40, color=DESTAQUE, edgecolors=FUNDO, linewidths=2, zorder=3)
    for indice in (0, len(x) - 1):
        eixo.annotate(
            numero(y[indice], 3 if indice else 2, textos["decimal"]), (x[indice], y[indice]),
            xytext=(8, 6), textcoords="offset points", fontsize=11, color=TINTA,
        )
    # Rótulos diretos no lugar da legenda, perto de onde cada linha fica visível.
    eixo.annotate(textos["serie_val"], (x[1], y[1]), xytext=(10, 22), textcoords="offset points",
                  fontsize=10, color=TINTA_SECUNDARIA)
    eixo.annotate(textos["serie_treino"], (xt[0], yt[0]), xytext=(-6, 8), textcoords="offset points",
                  fontsize=10, color=TINTA_APAGADA, ha="right")
    eixo.set_title(textos["curva_titulo"], loc="left", fontsize=12.5, color=TINTA, pad=12)
    eixo.set_xlabel(textos["curva_x"], fontsize=10)
    eixo.grid(axis="y", color=GRADE, linewidth=0.8)
    eixo.set_axisbelow(True)
    eixo.spines["left"].set_visible(False)
    eixo.tick_params(axis="y", length=0)
    eixo.set_ylim(0, 1.7)
    eixo.set_yticks([0, 0.5, 1.0, 1.5])
    eixo.yaxis.set_major_formatter(lambda v, _: numero(v, 1, textos["decimal"]))
    figura.tight_layout()
    figura.savefig(SAIDA / f"{idioma}-curva.png", dpi=200)
    plt.close(figura)


def main() -> None:
    estilo()
    SAIDA.mkdir(exist_ok=True)
    relatorios = {n: json.loads((RAIZ / "resultados" / f"{n}.json").read_text()) for n in ORDEM}
    for idioma in TEXTOS:
        resultados(relatorios, idioma)
        curva(idioma)
        print(f"figuras/{idioma}-resultados.png, figuras/{idioma}-curva.png")


if __name__ == "__main__":
    main()
