# Dados

`train.jsonl`, `valid.jsonl` e `test.jsonl` são amostras do **B2W-Reviews01**, da B2W Digital, reformatadas como conversas para o `mlx_lm lora`. São gerados por `uv run python -m avaliacoes.preparar`; as contagens, a semente e o SHA-256 da fonte estão em [`manifesto.json`](manifesto.json).

## Licença

O B2W-Reviews01 é distribuído pela B2W Digital sob a licença [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) ([repositório original](https://github.com/americanas-tech/b2w-reviews01)). Estes arquivos derivados seguem a mesma licença: uso não comercial, com crédito à B2W Digital e redistribuição sob os mesmos termos.

O espelho no Hugging Face de onde o Parquet é baixado ([ruanchaves/b2w-reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01)) marca o dataset como CC BY 4.0. Este projeto segue a licença do repositório original, que é a mais restritiva das duas.

## Referência

Livy Real, Marcio Oshiro e Alexandre Mafra. *B2W-Reviews01: An open product reviews corpus*. XII Symposium in Information and Human Language Technology (STIL), 2019.
