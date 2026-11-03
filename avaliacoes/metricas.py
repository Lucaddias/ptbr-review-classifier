"""Métricas da comparação, sem dependências: dá para conferir cada conta à mão."""

from __future__ import annotations

import random

NOTAS = (1, 2, 3, 4, 5)


def f1_macro(reais: list, previstos: list, classes: tuple) -> float:
    """Média simples do F1 de cada classe.

    É a métrica principal porque as notas são desbalanceadas (36% são 5): um
    modelo que chuta sempre 5 tem acurácia de 36% e F1 macro de ~11%.
    Previsão inválida (`None`) conta como erro para a classe real.
    """
    f1s = []
    for classe in classes:
        vp = sum(1 for r, p in zip(reais, previstos) if r == classe and p == classe)
        fp = sum(1 for r, p in zip(reais, previstos) if r != classe and p == classe)
        fn = sum(1 for r, p in zip(reais, previstos) if r == classe and p != classe)
        f1s.append(2 * vp / (2 * vp + fp + fn) if vp else 0.0)
    return sum(f1s) / len(f1s)


def matriz_de_confusao(reais: list[int], previstos: list[int | None]) -> list[list[int]]:
    """Linhas = nota real (1 a 5); colunas = prevista (1 a 5) e, por último, inválida."""
    matriz = [[0] * 6 for _ in NOTAS]
    for r, p in zip(reais, previstos):
        matriz[r - 1][(p - 1) if p is not None else 5] += 1
    return matriz


def resumir(reais_nota, previstos_nota, reais_rec, previstos_rec, formatos_validos) -> dict:
    n = len(reais_nota)
    validas = [(r, p) for r, p in zip(reais_nota, previstos_nota) if p is not None]
    return {
        "exemplos": n,
        "formato_valido": sum(formatos_validos) / n,
        "nota": {
            "acuracia": sum(r == p for r, p in zip(reais_nota, previstos_nota)) / n,
            "f1_macro": f1_macro(reais_nota, previstos_nota, NOTAS),
            # Erro médio absoluto e acerto com 1 de folga, só nas previsões
            # que trouxeram uma nota: inválida não tem distância.
            "erro_medio_absoluto": sum(abs(r - p) for r, p in validas) / len(validas) if validas else None,
            "acerto_com_folga_de_1": sum(abs(r - p) <= 1 for r, p in validas) / n,
            "matriz_de_confusao": matriz_de_confusao(reais_nota, previstos_nota),
        },
        "recomenda": {
            "acuracia": sum(r == p for r, p in zip(reais_rec, previstos_rec)) / n,
            "f1_macro": f1_macro(reais_rec, previstos_rec, (True, False)),
        },
    }


def intervalos_bootstrap(reais_nota, previstos_nota, reais_rec, previstos_rec, reamostras=1000, semente=0) -> dict:
    """Intervalo de 95% por bootstrap percentil, reamostrando os exemplos do teste.

    Com 1.000 exemplos e só 62 de nota 2, o F1 macro varia alguns pontos de
    uma amostra para outra; o intervalo diz se uma diferença entre modelos é
    maior que esse ruído.
    """
    gerador = random.Random(semente)
    n = len(reais_nota)
    f1s, acuracias = [], []
    for _ in range(reamostras):
        indices = [gerador.randrange(n) for _ in range(n)]
        f1s.append(f1_macro([reais_nota[i] for i in indices], [previstos_nota[i] for i in indices], NOTAS))
        acuracias.append(sum(reais_rec[i] == previstos_rec[i] for i in indices) / n)

    return {"nota_f1_macro": _percentis(f1s), "recomenda_acuracia": _percentis(acuracias)}


def diferenca_pareada(reais_nota, notas_a, notas_b, reais_rec, rec_a, rec_b, reamostras=1000, semente=0) -> dict:
    """Intervalo de 95% da diferença A − B, reamostrando os mesmos exemplos para os dois.

    Dois intervalos individuais que se encostam não dizem se A é melhor que B:
    os modelos erram nos mesmos exemplos difíceis, e a reamostragem pareada
    tira essa variação comum. Se o intervalo da diferença não contém zero, a
    diferença é maior que o ruído da amostra de teste.
    """
    gerador = random.Random(semente)
    n = len(reais_nota)
    dif_f1, dif_rec = [], []
    for _ in range(reamostras):
        indices = [gerador.randrange(n) for _ in range(n)]
        reais = [reais_nota[i] for i in indices]
        dif_f1.append(
            f1_macro(reais, [notas_a[i] for i in indices], NOTAS)
            - f1_macro(reais, [notas_b[i] for i in indices], NOTAS)
        )
        dif_rec.append(sum((reais_rec[i] == rec_a[i]) - (reais_rec[i] == rec_b[i]) for i in indices) / n)
    return {"nota_f1_macro": _percentis(dif_f1), "recomenda_acuracia": _percentis(dif_rec)}


def _percentis(valores: list[float]) -> list[float]:
    valores = sorted(valores)
    return [valores[int(0.025 * len(valores))], valores[int(0.975 * len(valores)) - 1]]
