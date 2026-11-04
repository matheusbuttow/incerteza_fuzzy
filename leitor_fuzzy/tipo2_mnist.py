"""
tipo2_mnist.py — fuzzy aditivo TIPO 2 com dados reais (5.000 dígitos do MNIST do mlxtend), classificando cada
resposta da rede em três zonas: ALARME (provável erro), SEGURO (provável acerto) e INCERTO.

Três redes diferentes (CNN, LeNet, MLP) são treinadas nos mesmos dados. As características vêm da biblioteca
(leitor_fuzzy); o tipo 2 é o de experimentos/ham_tipo2 (envelope entre redes, intervalo exato).
Teste "deixa uma rede fora": as regras são aprendidas em duas redes e aplicadas na terceira, que o sistema nunca viu.

    python -m leitor_fuzzy.tipo2_mnist
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1] / "experimentos" / "ham_tipo2"))
import fuzzy_t2 as t2  # noqa: E402

from leitor_fuzzy import LeitorFuzzy, treinar_rede  # noqa: E402
from leitor_fuzzy.leitor import CARACTERISTICAS  # noqa: E402


def redes():
    cnn = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))
    lenet = nn.Sequential(nn.Conv2d(1, 6, 5, padding=2), nn.Tanh(), nn.AvgPool2d(2), nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
                          nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10))
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    return {"CNN": cnn, "LeNet": lenet, "MLP": mlp}


def tabelas(X, y, semente, epocas=4):
    """Uma tabela de características por rede, nas MESMAS imagens de teste (divididas em duas metades)."""
    i = np.random.default_rng(semente).permutation(len(X)); tr, te = i[:2500], i[2500:]
    metade = np.random.default_rng(semente + 100).permutation(len(te)) % 2
    tabs = {}
    for nome, rede in redes().items():
        torch.manual_seed(semente)
        treinar_rede(rede, (X[tr], y[tr]), epocas=epocas, log=None)
        lf = LeitorFuzzy(rede, vistas="deslocamento").preparar((X[tr], y[tr]), log=lambda *a: None)
        F = lf.caracteristicas((X[te], y[te]))
        F["erro"], F["prox_p"], F["p"] = F.alucinou, 1 - F.confianca, F.confianca      # prox_p: maior = mais dúvida
        F["metade"], F["image_id"] = metade, te
        tabs[nome] = F
        print(f"  semente {semente} · {nome}: {len(lf.blocos)} blocos, acurácia {1 - F.erro.mean():.3f}, {F.erro.sum()} erros", flush=True)
    return tabs


def main(sementes=(0, 1, 2)):
    from mlxtend.data import mnist_data
    X, y = mnist_data()
    X = torch.tensor(X / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28); y = torch.tensor(y)
    cand = [c for c in CARACTERISTICAS if c != "p"] + ["prox_p"]
    L, Z, K = [], [], []
    for s in sementes:
        tabs = tabelas(X, y, s)
        lin, zon, regras, fous, preds = t2.deixa_uma_fora(tabs, cand, k_ad=4, n_boot=500, log=lambda *a: None)
        lin["semente"], zon["semente"] = s, s
        L.append(lin); Z.append(zon)
        if s == sementes[0]:
            K = regras
    L, Z = pd.concat(L), pd.concat(Z)
    pd.set_option("display.width", 250)
    mets = ["Só P", "Regressão logística transferida", "Fuzzy aditivo: tipo 1 dentro (referência)",
            "Fuzzy aditivo: tipo 1 transferido", "Fuzzy aditivo: tipo 2 transferido"]
    print("\nAUROC (média de", len(sementes), "sementes) — regras aprendidas nas OUTRAS duas redes:")
    print(L[L.método.isin(mets)].groupby(["método", "rede alvo"]).AUROC.mean().unstack().reindex(mets).round(3).to_string())
    t2r = L[L.método == "Fuzzy aditivo: tipo 2 transferido"]
    print("\nTipo 2 contra tipo 1 transferido (IC 95% por semente):")
    print(t2r[["semente", "rede alvo", "ganho de AUROC sobre o tipo 1 transferido", "IC 95% do ganho"]].round(3).to_string(index=False))
    z = Z[Z.sistema == "Fuzzy aditivo tipo 2"]
    print("\nTrês zonas (fuzzy aditivo tipo 2), média das sementes:")
    print(z.groupby(["rede alvo", "zona"])[["% das imagens", "taxa de erro da rede na zona", "% dos erros"]].mean()
          .round(3).to_string())
    print("\nRegras (alvo CNN, aprendidas em LeNet + MLP, semente", sementes[0], "):")
    print(K[("CNN", "Fuzzy aditivo")].to_string(index=False))
    return L, Z


if __name__ == "__main__":
    main()
