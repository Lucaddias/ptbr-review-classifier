---
license: cc-by-nc-sa-4.0
language:
- pt
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
library_name: mlx
datasets:
- ruanchaves/b2w-reviews01
pipeline_tag: text-generation
tags:
- mlx
- lora
- sentiment-analysis
- text-classification
- reviews
---

# Qwen3.5-0.8B · avaliações de clientes (LoRA)

Adaptador LoRA de 14 MB para o [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), treinado com [mlx-lm](https://github.com/ml-explore/mlx-lm). Lê uma avaliação de cliente em português e devolve, em JSON, a nota que a pessoa deu ao produto (1 a 5) e se ela o recomendaria a um amigo.

```text
Título: Não recomendo
Avaliação: Chegou com a tela trincada e o suporte não responde.
→ {"nota": 1, "recomenda": false}
```

Código, dados e método: [github.com/Lucaddias/ptbr-review-classifier](https://github.com/Lucaddias/ptbr-review-classifier).

## Resultados

1.000 avaliações de teste do B2W-Reviews01, de produtos que não aparecem no treino. Intervalos de 95% por bootstrap.

| Modelo | Formato válido | Nota: F1 macro | Nota: acurácia | Nota: erro ≤ 1 | Recomenda: acurácia |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B, sem ajuste | 96,5% | 0,437 (0,403–0,470) | 55,3% | 88,1% | 82,7% |
| **Este adaptador** | **100%** | **0,503** (0,470–0,535) | 61,2% | **94,8%** | **92,1%** |
| Qwen3.5-9B Q4_K_M, sem ajuste | 99,6% | 0,502 (0,466–0,538) | **63,9%** | 92,5% | 90,2% |

Em diferença pareada, o adaptador empata com o 9B no F1 macro da nota (+0,001, IC de −0,041 a +0,040) e o supera na recomendação (+1,9 ponto, IC de +0,6 a +3,2). Uma avaliação por vez, num Mac M5, responde em 0,26 s; o 9B no llama.cpp, em 1,02 s.

## Como usar

Só no MLX (Mac com Apple Silicon). Este repositório tem só o adaptador, então o modelo base é carregado à parte; o exemplo automático do botão "Use this model", que carrega este repositório sozinho, não funciona. O prompt precisa ser o mesmo do treino:

```python
from huggingface_hub import snapshot_download
from mlx_lm import generate, load

adaptador = snapshot_download("oluca/qwen3.5-0.8b-ptbr-reviews-lora")
modelo, tokenizador = load("Qwen/Qwen3.5-0.8B", adapter_path=adaptador)

titulo = "Bom, mas demorou"
texto = "O fone é ótimo, som limpo e bateria boa. Só a entrega que atrasou uma semana."
mensagens = [
    {"role": "system", "content": "Você classifica avaliações de clientes de uma loja online brasileira."},
    {"role": "user", "content": (
        "Leia a avaliação e responda só com um JSON no formato "
        '{"nota": <inteiro de 1 a 5>, "recomenda": <true ou false>}, '
        "onde nota é a nota que o cliente deu ao produto e recomenda diz se ele "
        "recomendaria o produto a um amigo."
        f"\n\nTítulo: {titulo}\nAvaliação: {texto}"
    )},
]
prompt = tokenizador.apply_chat_template(mensagens, add_generation_prompt=True, enable_thinking=False)
print(generate(modelo, tokenizador, prompt, max_tokens=24))
# {"nota": 3, "recomenda": true}
```

## Treino

- **Dados:** 4.000 avaliações do [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) (Americanas, 2018), sem textos repetidos, com a divisão entre treino, validação e teste feita por produto.
- **LoRA:** rank 16, escala 20, dropout 0,05, nas camadas lineares das 8 camadas de cima. As 16 de baixo ficam congeladas e em modo de avaliação, para usar o kernel Metal da atenção linear (detalhes no repositório).
- **Otimização:** 500 atualizações de 8 exemplos (uma passada), AdamW, taxa de 1e-4 com aquecimento de 25 passos e decaimento cosseno, perda só nos tokens da resposta.
- **Custo:** 45 minutos num Mac M5 de 24 GB, com pico de 17 GB. Perda de validação de 1,52 antes do treino e de 0,050 no fim.

## Limites

- Treinado e testado em avaliações de uma loja, de janeiro a maio de 2018.
- Erra mais a nota 2: acerta 8 de 62, e manda 41 delas para nota 1. A nota 2 é só 6% do treino.
- Os rótulos do dataset têm ruído: perguntas postadas como avaliação com nota 5, elogios ao produto com reclamação da entrega.
- Um único treino, sem busca de hiperparâmetros nem repetição com outras sementes.

## Licença

O B2W-Reviews01 é distribuído pela B2W Digital sob CC BY-NC-SA 4.0, e este adaptador, treinado com ele, segue a mesma licença: uso não comercial, com crédito e sob os mesmos termos. O espelho no Hugging Face marca o dataset como CC BY 4.0; aqui vale a licença do repositório original. O Qwen3.5-0.8B é Apache 2.0.

## Referência do dataset

```bibtex
@inproceedings{real2019b2w,
  title={B2W-reviews01: an open product reviews corpus},
  author={Real, Livy and Oshiro, Marcio and Mafra, Alexandre},
  booktitle={STIL-Symposium in Information and Human Language Technology},
  year={2019}
}
```
