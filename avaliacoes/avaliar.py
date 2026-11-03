"""Roda um modelo no conjunto de teste e grava as métricas em resultados/.

    # Qwen3.5-0.8B sem ajuste, e com o adaptador treinado:
    uv run python -m avaliacoes.avaliar --nome base-0.8b
    uv run python -m avaliacoes.avaliar --nome lora-0.8b --adaptador adaptador

    # Qwen3.5-9B quantizado, servido pelo llama-server do llama.cpp:
    llama-server -m Qwen_Qwen3.5-9B-Q4_K_M.gguf --jinja --port 8089
    uv run python -m avaliacoes.avaliar --nome qwen3.5-9b-q4 --servidor http://127.0.0.1:8089

Decodificação gulosa e pensamento desligado nos três: a mesma pergunta, a
resposta mais provável de cada modelo.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from avaliacoes import metricas, prompt

RAIZ = Path(__file__).resolve().parent.parent
MODELO_BASE = "Qwen/Qwen3.5-0.8B"
REVISAO_BASE = "2fc06364715b967f1860aea9cf38778875588b17"
# A resposta certa tem 12 tokens; folga para o modelo base, que às vezes explica.
MAXIMO_DE_TOKENS = 48


def carregar_teste(caminho: Path, limite: int | None) -> list[dict]:
    exemplos = []
    for linha in caminho.read_text(encoding="utf-8").splitlines()[:limite]:
        mensagens = json.loads(linha)["messages"]
        nota, recomenda, _ = prompt.interpretar(mensagens[-1]["content"])
        exemplos.append({"mensagens": mensagens[:-1], "nota": nota, "recomenda": recomenda})
    return exemplos


def gerar_com_mlx(exemplos: list[dict], adaptador: str | None, lote: int) -> list[str]:
    from mlx_lm import batch_generate, load

    modelo, tokenizador = load(MODELO_BASE, revision=REVISAO_BASE, adapter_path=adaptador)
    prompts = [
        tokenizador.apply_chat_template(e["mensagens"], add_generation_prompt=True, enable_thinking=False)
        for e in exemplos
    ]
    saidas = []
    for inicio in range(0, len(prompts), lote):
        resposta = batch_generate(modelo, tokenizador, prompts[inicio:inicio + lote], max_tokens=MAXIMO_DE_TOKENS)
        saidas.extend(resposta.texts)
        print(f"  {len(saidas)}/{len(prompts)}", flush=True)
    return saidas


def modelo_do_servidor(servidor: str) -> str:
    """Nome do arquivo GGUF que o llama-server carregou, sem o caminho local."""
    with urllib.request.urlopen(f"{servidor}/v1/models", timeout=10) as resposta:
        return f"{Path(json.load(resposta)['data'][0]['id']).name} via llama-server"


def gerar_com_servidor(exemplos: list[dict], servidor: str) -> list[str]:
    saidas = []
    for e in exemplos:
        corpo = json.dumps({
            "messages": e["mensagens"],
            "temperature": 0,
            "max_tokens": MAXIMO_DE_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        pedido = urllib.request.Request(
            f"{servidor}/v1/chat/completions", corpo, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(pedido, timeout=120) as resposta:
            saidas.append(json.load(resposta)["choices"][0]["message"]["content"])
        if len(saidas) % 100 == 0:
            print(f"  {len(saidas)}/{len(exemplos)}", flush=True)
    return saidas


def main() -> None:
    argumentos = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argumentos.add_argument("--nome", required=True, help="nome do arquivo em resultados/")
    argumentos.add_argument("--adaptador", help="pasta do adaptador LoRA (só no modo mlx)")
    argumentos.add_argument("--servidor", help="URL de um llama-server em vez do mlx")
    argumentos.add_argument("--dados", type=Path, default=RAIZ / "dados" / "test.jsonl")
    argumentos.add_argument("--limite", type=int, help="avalia só os N primeiros exemplos")
    argumentos.add_argument("--lote", type=int, default=32)
    opcoes = argumentos.parse_args()

    exemplos = carregar_teste(opcoes.dados, opcoes.limite)
    inicio = time.perf_counter()
    if opcoes.servidor:
        saidas = gerar_com_servidor(exemplos, opcoes.servidor)
    else:
        saidas = gerar_com_mlx(exemplos, opcoes.adaptador, opcoes.lote)
    segundos = time.perf_counter() - inicio

    lidas = [prompt.interpretar(s) for s in saidas]
    resumo = metricas.resumir(
        [e["nota"] for e in exemplos],
        [nota for nota, _, _ in lidas],
        [e["recomenda"] for e in exemplos],
        [recomenda for _, recomenda, _ in lidas],
        [valido for _, _, valido in lidas],
    )
    resumo["intervalos_95"] = metricas.intervalos_bootstrap(
        [e["nota"] for e in exemplos], [nota for nota, _, _ in lidas],
        [e["recomenda"] for e in exemplos], [recomenda for _, recomenda, _ in lidas],
    )
    relatorio = {
        "nome": opcoes.nome,
        "modelo": modelo_do_servidor(opcoes.servidor) if opcoes.servidor else f"{MODELO_BASE}@{REVISAO_BASE}",
        "adaptador": opcoes.adaptador,
        "segundos": round(segundos, 1),
        "exemplos_por_segundo": round(len(exemplos) / segundos, 2),
        "metricas": resumo,
        "previsoes": [
            {"saida": s, "nota": e["nota"], "recomenda": e["recomenda"], "nota_prevista": n, "recomenda_prevista": r}
            for s, e, (n, r, _) in zip(saidas, exemplos, lidas)
        ],
    }
    destino = RAIZ / "resultados" / f"{opcoes.nome}.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(relatorio, ensure_ascii=False, indent=1) + "\n")

    nota, recomenda = resumo["nota"], resumo["recomenda"]
    ic = resumo["intervalos_95"]
    print(f"\n{opcoes.nome}: {len(exemplos)} exemplos em {segundos:.0f} s")
    print(f"  formato válido      {resumo['formato_valido']:.1%}")
    print(f"  nota: F1 macro      {nota['f1_macro']:.3f}  (IC 95% {ic['nota_f1_macro'][0]:.3f}–{ic['nota_f1_macro'][1]:.3f})")
    print(f"  nota: acurácia      {nota['acuracia']:.1%}   ±1: {nota['acerto_com_folga_de_1']:.1%}   EMA: {nota['erro_medio_absoluto']}")
    print(f"  recomenda: acurácia {recomenda['acuracia']:.1%}   F1 macro {recomenda['f1_macro']:.3f}")
    print(f"  gravado em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
