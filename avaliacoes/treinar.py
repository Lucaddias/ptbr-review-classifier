"""Treina o adaptador LoRA com config/lora.yaml.

    uv run python -m avaliacoes.treinar [--config outro.yaml]

Chama o treino do mlx-lm dentro deste processo, e não pela linha de comando,
por três motivos: o `mlx_lm lora` não aceita revisão do Hugging Face (aqui o
modelo vem da revisão fixada), fora de um terminal ele só escreve o log no
fim, e as camadas congeladas precisam ficar em modo de avaliação (ver
`congelar_em_avaliacao`). O callback grava cada perda em
resultados/curva-de-treino.jsonl assim que ela sai.
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from mlx_lm.lora import CONFIG_DEFAULTS, yaml_loader
from mlx_lm.lora import train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.utils import load

from avaliacoes.avaliar import MODELO_BASE, RAIZ, REVISAO_BASE


class Curva(TrainingCallback):
    def __init__(self, caminho: Path):
        self.arquivo = caminho.open("w")
        self.inicio = time.perf_counter()

    def _gravar(self, tipo: str, info: dict) -> None:
        linha = {"tipo": tipo, "segundos": round(time.perf_counter() - self.inicio, 1), **info}
        self.arquivo.write(json.dumps(linha) + "\n")
        self.arquivo.flush()
        print(json.dumps(linha), flush=True)

    def on_train_loss_report(self, train_info: dict):
        self._gravar("treino", train_info)

    def on_val_loss_report(self, val_info: dict):
        self._gravar("validacao", val_info)


def congelar_em_avaliacao(camadas) -> None:
    """Mantém estas camadas em modo de avaliação mesmo durante o treino.

    18 das 24 camadas do Qwen3.5 são de atenção linear (Gated DeltaNet). No
    mlx-lm elas só usam o kernel Metal fora do treino; em modo de treino caem
    num laço em Python, um token por vez, que guarda o estado de cada passo
    para o backward: 0,13 lote por segundo e 12 GB de pico com LoRA nas 24.
    O kernel não tem gradiente, mas as camadas abaixo do LoRA não precisam
    de um: nada treinável fica antes delas.
    """
    def fixar(_, modulo):
        modulo._training = False
        modulo._set_training_mode = lambda modo: None

    for camada in camadas:
        camada.apply_to_modules(fixar)


def main() -> None:
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("--config", type=Path, default=RAIZ / "config" / "lora.yaml")
    argumentos.add_argument("--curva", type=Path, default=RAIZ / "resultados" / "curva-de-treino.jsonl")
    opcoes = argumentos.parse_args()

    config = {**CONFIG_DEFAULTS, **yaml.load(opcoes.config.read_text(), yaml_loader)}
    config["model"] = snapshot_download(MODELO_BASE, revision=REVISAO_BASE)
    args = types.SimpleNamespace(**config)

    modelo, tokenizador = load(args.model)
    if args.num_layers > 0:
        congelar_em_avaliacao(modelo.layers[: -args.num_layers])
    treino, validacao, _ = load_dataset(args, tokenizador)
    train_model(args, modelo, treino, validacao, Curva(opcoes.curva))


if __name__ == "__main__":
    main()
