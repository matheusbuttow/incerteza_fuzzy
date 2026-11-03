"""
leitor_fuzzy — lê uma rede neural congelada por dentro e diz quando ela provavelmente alucinou, e por quê.

    from leitor_fuzzy import LeitorFuzzy, treinar_rede
    lf = LeitorFuzzy(rede, vistas="imagem").fit(dados_referencia, dados_calibracao)
    risco = lf.risco(dados_teste)
    print(lf.diagnostico(dados_teste, nomes_classes))
"""
from .blocos import achar_blocos
from .leitor import CARACTERISTICAS, LeitorFuzzy
from .treino import treinar_rede

__all__ = ["LeitorFuzzy", "treinar_rede", "achar_blocos", "CARACTERISTICAS"]
