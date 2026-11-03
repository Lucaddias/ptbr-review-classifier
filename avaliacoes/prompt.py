"""O prompt e o formato de resposta, iguais para todos os modelos comparados.

Um prompt só para o modelo base, o ajustado e o 9B: a comparação mede o
modelo, não a engenharia de prompt de cada um.
"""

from __future__ import annotations

import json
import re

SISTEMA = "Você classifica avaliações de clientes de uma loja online brasileira."

INSTRUCAO = (
    "Leia a avaliação e responda só com um JSON no formato "
    '{"nota": <inteiro de 1 a 5>, "recomenda": <true ou false>}, '
    "onde nota é a nota que o cliente deu ao produto e recomenda diz se ele "
    "recomendaria o produto a um amigo."
)


def mensagens(titulo: str, texto: str) -> list[dict]:
    corpo = f"{INSTRUCAO}\n\nTítulo: {titulo}\nAvaliação: {texto}"
    return [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": corpo},
    ]


def resposta(nota: int, recomenda: bool) -> str:
    return json.dumps({"nota": nota, "recomenda": recomenda})


def interpretar(saida: str) -> tuple[int | None, bool | None, bool]:
    """Lê a resposta do modelo: `(nota, recomenda, formato_valido)`.

    `formato_valido` só é verdadeiro para um JSON com os dois campos nos tipos
    pedidos. Fora disso, ainda tenta resgatar os valores por expressão regular:
    um modelo que responde "nota: 4" acertou a nota, só errou o formato, e as
    duas coisas são medidas separadamente.
    """
    trecho = re.search(r"\{.*?\}", saida, flags=re.DOTALL)
    if trecho:
        try:
            objeto = json.loads(trecho.group(0))
            nota, recomenda = objeto.get("nota"), objeto.get("recomenda")
            if isinstance(nota, int) and not isinstance(nota, bool) and 1 <= nota <= 5 and isinstance(recomenda, bool):
                return nota, recomenda, True
        except (json.JSONDecodeError, AttributeError):
            pass

    achou_nota = re.search(r"nota\W{0,4}([1-5])\b", saida, flags=re.IGNORECASE)
    achou_recomenda = re.search(r"recomenda\W{0,4}(true|false|sim|n[ãa]o)\b", saida, flags=re.IGNORECASE)
    nota = int(achou_nota.group(1)) if achou_nota else None
    recomenda = achou_recomenda.group(1).lower() in ("true", "sim") if achou_recomenda else None
    return nota, recomenda, False
