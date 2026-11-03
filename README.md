# ptbr-review-classifier

Um adaptador LoRA de 14 MB para o Qwen3.5-0.8B que lê uma avaliação de cliente em português e devolve, em JSON, a nota que a pessoa deu (1 a 5) e se ela recomendaria o produto:

```text
Título: Não recomendo
Avaliação: Chegou com a tela trincada e o suporte não responde.
→ {"nota": 1, "recomenda": false}
```

O treino roda num Mac com [mlx-lm](https://github.com/ml-explore/mlx-lm), sobre o [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) e o [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) (avaliações reais da Americanas). A pergunta do projeto: **um modelo pequeno, ajustado para uma tarefa, chega perto de um modelo 11 vezes maior usado sem ajuste?**

Na nota, empata com o Qwen3.5-9B. Na recomendação, passa. E, no mesmo Mac, responde quase quatro vezes mais rápido.

O adaptador está no Hugging Face: [oluca/qwen3.5-0.8b-ptbr-reviews-lora](https://huggingface.co/oluca/qwen3.5-0.8b-ptbr-reviews-lora), com o exemplo de uso no model card. A história do projeto, com o que deu errado no caminho, está no artigo: [português, no Medium](https://medium.com/@lucadiasferreira/ajustei-um-modelo-de-0-8b-e-ele-empatou-com-um-de-9b-veja-como-medi-6bf09b57f9fd) · [English](artigo/en.md).

## Resultados

1.000 avaliações de teste, de produtos que não aparecem no treino. Intervalos de 95% por bootstrap.

| Modelo | Pesos | Formato válido | Nota: F1 macro | Nota: acurácia | Nota: erro ≤ 1 | Recomenda: acurácia |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B, sem ajuste | 1,7 GB | 96,5% | 0,437 (0,403–0,470) | 55,3% | 88,1% | 82,7% (80,3–85,0) |
| **Qwen3.5-0.8B + LoRA** | 1,7 GB + 14 MB | **100%** | **0,503** (0,470–0,535) | 61,2% | **94,8%** | **92,1%** (90,4–93,7) |
| Qwen3.5-9B Q4_K_M, sem ajuste | 6,2 GB | 99,6% | 0,502 (0,466–0,538) | **63,9%** | 92,5% | 90,2% (88,3–91,9) |
| *Referência: sempre 5 / sempre recomenda* | | | *0,115* | *40,3%* | | *76,4%* |

Diferenças pareadas (mesmas reamostragens para os dois modelos):

| Comparação | F1 macro da nota | Acurácia de recomendação |
|---|---:|---:|
| 0.8B + LoRA − 0.8B sem ajuste | +0,066 (+0,024 a +0,109) | +9,4 pontos (+7,5 a +11,4) |
| 0.8B + LoRA − 9B | +0,001 (−0,041 a +0,040) | +1,9 ponto (+0,6 a +3,2) |

Uma avaliação por vez, no mesmo Mac (M5, 24 GB), o 0.8B ajustado responde em 0,26 s e o 9B em 1,02 s. Em lotes de 32, o 0.8B processa 15 avaliações por segundo. O treino levou 45 minutos, com pico de 17 GB de memória.

### O que os números mostram

- **Na nota, o ajuste levou o 0.8B ao nível do 9B.** A diferença de F1 macro entre os dois é de 0,001, bem dentro do ruído da amostra. Na recomendação, o 0.8B ajustado é melhor por 1,9 ponto, e o intervalo não chega a zero.
- **O modelo base chuta 5 demais e às vezes erra o formato.** 170 das 224 notas 4 viraram 5, e 35 respostas fogem do formato: `"nota": 0` (18), `"recomenda": "true"` como texto (14) ou `"nota": 10` (3). Depois do ajuste, o formato não falha nenhuma vez, e o acerto nas notas 4 sobe de 33 para 97.
- **A acurácia exata favorece o 9B porque ele também responde muito 5.** Ele acerta 386 das 403 notas 5, mas só 44 das 224 notas 4. Como 40% do teste é nota 5, isso rende acurácia. O F1 macro pesa as cinco notas igualmente e não premia esse viés.
- **A nota 2 piorou com o ajuste.** O modelo base acerta 22 das 62 notas 2; o ajustado, 8, e manda 41 delas para nota 1. A nota 2 é só 6% do treino, e o adaptador aprendeu essa raridade. Amostragem balanceada no treino é o próximo experimento.
- **Na recomendação, o ajustado chega perto do que a própria nota explica.** A regra "recomenda se a nota real for 3 ou mais" acerta 95,4% do teste. O modelo, lendo só o texto, chega a 92,1%.

## Método

**Dados.** O B2W-Reviews01 tem 132.373 avaliações. Saem as sem texto (3.275), as sem resposta de recomendação (18) e os textos repetidos (2.384, comparados sem diferença de caixa e espaços). A divisão entre treino, validação e teste é **por produto**, pelo hash do `product_id`: todas as avaliações de um produto ficam na mesma parte. Dividir por avaliação deixaria o modelo ver no treino outras opiniões sobre o mesmo item do teste. De cada parte sai uma amostra aleatória com semente fixa (4.000 de treino, 400 de validação, 1.000 de teste), sem balancear: o teste tem a distribuição real das notas, com 40% de notas 5 e 6% de notas 2. Contagens em [`dados/manifesto.json`](dados/manifesto.json).

**Prompt.** O mesmo para os três modelos, em [`avaliacoes/prompt.py`](avaliacoes/prompt.py): uma instrução de sistema, o pedido do formato JSON, o título e o texto. Decodificação gulosa e modo de pensamento desligado.

**Treino.** LoRA de rank 16 nas camadas lineares das 8 camadas de cima, perda só nos tokens da resposta (`mask_prompt`), 500 atualizações de 8 exemplos (uma passada pelo treino), AdamW com taxa de 1e-4 e decaimento cosseno. Configuração em [`config/lora.yaml`](config/lora.yaml).

**Por que só 8 camadas.** 18 das 24 camadas do Qwen3.5 são de atenção linear (Gated DeltaNet). No mlx-lm, essas camadas só usam o kernel Metal fora do treino; em modo de treino, caem num laço em Python que processa um token por vez e guarda o estado de cada passo para o backward. Com LoRA nas 24 camadas, o treino rodou a 0,13 lote de 4 por segundo, com 12 GB de pico, e lotes de 8 estouraram a memória da GPU num Mac de 24 GB. Como nada treinável fica abaixo das camadas congeladas, elas não precisam de gradiente: [`avaliacoes/treinar.py`](avaliacoes/treinar.py) as mantém em modo de avaliação, com o kernel rápido, e o laço lento fica só nas 6 camadas de atenção linear que recebem LoRA. Resultado: ~0,37 lote por segundo, quase três vezes mais rápido.

**Métricas.** A principal é o **F1 macro da nota**, a média do F1 de cada nota: com notas desbalanceadas, um modelo que responde sempre 5 tem 40% de acurácia e só 11% de F1 macro. O intervalo de 95% vem de 1.000 reamostragens bootstrap do teste. Formato válido conta só JSON com os dois campos nos tipos certos; quando o formato falha, os valores ainda são lidos por expressão regular, para separar "errou o formato" de "errou a nota". Código e testes em [`avaliacoes/metricas.py`](avaliacoes/metricas.py) e [`tests/`](tests/).

## Limites

- **Uma loja, um período.** São avaliações da Americanas de janeiro a maio de 2018. O intervalo de confiança mede a incerteza da amostra, não o quanto o resultado vale para outros sites ou outros anos.
- **Os rótulos têm ruído.** Dos 12 erros de 3 pontos ou mais do modelo ajustado, a maioria é de perguntas postadas como avaliação, como "Esse Hoverboard são de quantas polegadas?" com nota 5, ou de textos que elogiam o produto e reclamam da entrega.
- **Um treino só.** Uma semente e uma configuração, sem busca de hiperparâmetros; a variação entre treinos não está medida.
- **Os modelos sem ajuste foram testados sem exemplos no prompt.** Um prompt com alguns exemplos poderia melhorar o base e o 9B.
- **A comparação de velocidade é entre conjuntos, não só entre modelos.** O 0.8B roda no MLX em bf16; o 9B, no llama.cpp com 4 bits. A quantização também pode custar alguma qualidade ao 9B.
- **Gerar em lote mudou 1 de 200 respostas** em relação a gerar uma por vez (uma nota 1 virou 2), por efeito numérico do preenchimento dos lotes.

## Como reproduzir

Precisa de um Mac com Apple Silicon e do [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m avaliacoes.preparar        # baixa o dataset (SHA-256 conferido) e gera dados/*.jsonl
uv run python -m avaliacoes.avaliar --nome base-0.8b
uv run python -m avaliacoes.treinar         # grava o adaptador em adaptador/
uv run python -m avaliacoes.avaliar --nome lora-0.8b --adaptador adaptador
uv run pytest
```

Para o 9B, com o [llama.cpp](https://github.com/ggml-org/llama.cpp) instalado e o GGUF do [Qwen3.5-9B Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF):

```bash
llama-server -m Qwen_Qwen3.5-9B-Q4_K_M.gguf --jinja --port 8089
uv run python -m avaliacoes.avaliar --nome qwen3.5-9b-q4 --servidor http://127.0.0.1:8089
uv run python -m avaliacoes.comparar base-0.8b lora-0.8b qwen3.5-9b-q4
```

## Licença

Código sob a licença MIT. Os dados em `dados/` e o adaptador treinado derivam do B2W-Reviews01, publicado pela B2W Digital sob [CC BY-NC-SA 4.0](https://github.com/americanas-tech/b2w-reviews01), e seguem a mesma licença: uso não comercial, com crédito e sob os mesmos termos. O espelho do dataset no Hugging Face o marca como CC BY 4.0; aqui vale a licença do repositório original. Detalhes em [`dados/README.md`](dados/README.md).
