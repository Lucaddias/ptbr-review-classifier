"""Monta as tabelas do README a partir dos relatórios em resultados/.

    uv run python -m avaliacoes.comparar base-0.8b lora-0.8b qwen3.5-9b-q4

A primeira tabela traz as métricas de cada modelo; a segunda, a diferença de
cada par com intervalo de 95% por bootstrap pareado.
"""

from __future__ import annotations

import itertools
import json
import sys

from avaliacoes import metricas
from avaliacoes.avaliar import RAIZ


def carregar(nome: str) -> dict:
    return json.loads((RAIZ / "resultados" / f"{nome}.json").read_text())


def main() -> None:
    relatorios = {nome: carregar(nome) for nome in sys.argv[1:]}

    print("| Modelo | Formato válido | Nota: F1 macro (IC 95%) | Nota: acurácia | Nota: ±1 | Recomenda: acurácia (IC 95%) | Exemplos/s |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for nome, r in relatorios.items():
        m = r["metricas"]
        f1, rec = m["intervalos_95"]["nota_f1_macro"], m["intervalos_95"]["recomenda_acuracia"]
        print(
            f"| {nome} | {m['formato_valido']:.1%} "
            f"| {m['nota']['f1_macro']:.3f} ({f1[0]:.3f}–{f1[1]:.3f}) "
            f"| {m['nota']['acuracia']:.1%} | {m['nota']['acerto_com_folga_de_1']:.1%} "
            f"| {m['recomenda']['acuracia']:.1%} ({rec[0]:.1%}–{rec[1]:.1%}) | {r['exemplos_por_segundo']:.1f} |"
        )

    print()
    print("| Comparação | Diferença no F1 macro da nota (IC 95%) | Diferença na acurácia de recomendação (IC 95%) |")
    print("|---|---:|---:|")
    for a, b in itertools.combinations(relatorios, 2):
        pa, pb = relatorios[a]["previsoes"], relatorios[b]["previsoes"]
        assert [p["nota"] for p in pa] == [p["nota"] for p in pb], "relatórios de testes diferentes"
        dif = metricas.diferenca_pareada(
            [p["nota"] for p in pa], [p["nota_prevista"] for p in pb], [p["nota_prevista"] for p in pa],
            [p["recomenda"] for p in pa], [p["recomenda_prevista"] for p in pb], [p["recomenda_prevista"] for p in pa],
        )
        ma, mb = relatorios[a]["metricas"], relatorios[b]["metricas"]
        d_f1 = mb["nota"]["f1_macro"] - ma["nota"]["f1_macro"]
        d_rec = mb["recomenda"]["acuracia"] - ma["recomenda"]["acuracia"]
        f1, rec = dif["nota_f1_macro"], dif["recomenda_acuracia"]
        print(
            f"| {b} − {a} | {d_f1:+.3f} ({f1[0]:+.3f} a {f1[1]:+.3f}) "
            f"| {d_rec:+.1%} ({rec[0]:+.1%} a {rec[1]:+.1%}) |"
        )


if __name__ == "__main__":
    main()
