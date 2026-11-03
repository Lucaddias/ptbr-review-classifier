import mlx.nn as nn

from avaliacoes.treinar import congelar_em_avaliacao


class Bloco(nn.Module):
    def __init__(self):
        super().__init__()
        self.interna = nn.Linear(2, 2)
        self.queda = nn.Dropout(0.5)


def test_camadas_congeladas_ignoram_model_train():
    camadas = [Bloco() for _ in range(3)]
    modelo = nn.Sequential(*camadas)
    congelar_em_avaliacao(camadas[:2])

    modelo.train()

    assert not camadas[0].training and not camadas[0].interna.training and not camadas[1].queda.training
    assert camadas[2].training and camadas[2].queda.training
