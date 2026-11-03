from avaliacoes import metricas, prompt


def test_f1_macro_conta_cada_classe_igual():
    # Classe 1: vp=1, fp=0, fn=1 -> 2/3. Classe 2: vp=1, fp=1, fn=0 -> 2/3.
    assert metricas.f1_macro([1, 1, 2], [1, 2, 2], (1, 2)) == 2 / 3


def test_f1_macro_chutar_sempre_a_maioria_pontua_pouco():
    reais = [5] * 36 + [4] * 24 + [1] * 18 + [3] * 13 + [2] * 9
    f1 = metricas.f1_macro(reais, [5] * len(reais), metricas.NOTAS)
    assert round(f1, 3) == round((2 * 36 / (2 * 36 + 64)) / 5, 3)


def test_previsao_invalida_conta_como_erro():
    assert metricas.f1_macro([1, 2], [None, 2], (1, 2)) == (0 + 1) / 2


def test_matriz_de_confusao_poe_invalidas_na_ultima_coluna():
    matriz = metricas.matriz_de_confusao([1, 5, 5], [1, 4, None])
    assert matriz[0] == [1, 0, 0, 0, 0, 0]
    assert matriz[4] == [0, 0, 0, 1, 0, 1]


def test_resumir_separa_formato_de_acerto():
    resumo = metricas.resumir([4, 2], [4, 3], [True, False], [True, True], [True, False])
    assert resumo["formato_valido"] == 0.5
    assert resumo["nota"]["acuracia"] == 0.5
    assert resumo["nota"]["erro_medio_absoluto"] == 0.5
    assert resumo["nota"]["acerto_com_folga_de_1"] == 1.0
    assert resumo["recomenda"]["acuracia"] == 0.5


def test_diferenca_pareada_de_modelos_iguais_e_zero():
    reais = [1, 2, 3, 4, 5] * 20
    previstos = [1, 2, 3, 5, 5] * 20
    rec = [True, False] * 50
    dif = metricas.diferenca_pareada(reais, previstos, previstos, rec, rec, rec, reamostras=200)
    assert dif == {"nota_f1_macro": [0.0, 0.0], "recomenda_acuracia": [0.0, 0.0]}


def test_diferenca_pareada_detecta_modelo_melhor():
    reais = [1, 2, 3, 4, 5] * 40
    perfeito = list(reais)
    chuta_cinco = [5] * len(reais)
    rec = [True] * len(reais)
    dif = metricas.diferenca_pareada(reais, perfeito, chuta_cinco, rec, rec, rec, reamostras=200)
    assert dif["nota_f1_macro"][0] > 0.5


def test_interpretar_json_valido():
    assert prompt.interpretar('{"nota": 4, "recomenda": false}') == (4, False, True)


def test_interpretar_json_cercado_de_texto():
    saida = 'Claro! ```json\n{"nota": 2, "recomenda": false}\n```'
    assert prompt.interpretar(saida) == (2, False, True)


def test_interpretar_resgata_valores_fora_do_formato():
    assert prompt.interpretar("nota: 5, recomenda: sim") == (5, True, False)


def test_interpretar_rejeita_tipos_errados():
    # "4" como texto e nota fora da escala não valem como formato correto.
    assert prompt.interpretar('{"nota": "4", "recomenda": true}')[2] is False
    assert prompt.interpretar('{"nota": 7, "recomenda": true}') == (None, True, False)


def test_resposta_e_o_que_interpretar_le():
    assert prompt.interpretar(prompt.resposta(3, True)) == (3, True, True)
