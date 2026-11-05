"""
teste_fashion.py — teste com dados reais mais difíceis: Fashion-MNIST (70.000 fotos de roupas, 10 classes).
Três redes (CNN, LeNet, MLP), cada uma lida pela biblioteca; grava os mesmos arquivos do pipeline principal
(features.csv.gz e escores.csv.gz), para rodar depois o analise_confiantes.py em cima deles.

    python -m leitor_fuzzy.teste_fashion --dados fashion_mnist.npz --saida resultados_fashion
    python analise_confiantes.py --pasta resultados_fashion --frac 0.2 0.3 0.5

O .npz tem X (n, 28, 28) uint8 e y (n,). Espelhar uma roupa não muda a classe: vista = espelho horizontal.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import comparacao as cmp
import fuzzy as fz
from leitor_fuzzy import LeitorFuzzy, treinar_rede
from leitor_fuzzy.tipo2_mnist import redes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--saida", default="resultados_fashion")
    ap.add_argument("--sementes", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-treino", type=int, default=20000)
    ap.add_argument("--n-teste", type=int, default=10000)
    ap.add_argument("--epocas", type=int, default=6)
    a = ap.parse_args()
    d = np.load(a.dados)
    X = torch.tensor(d["X"] / 255.0, dtype=torch.float32)[:, None]; y = torch.tensor(d["y"])
    nossas = [c for c in cmp.NOSSAS]
    L = []
    for s in a.sementes:
        i = np.random.default_rng(s).permutation(len(X)); tr, te = i[:a.n_treino], i[a.n_treino:a.n_treino + a.n_teste]
        ref = tr[:5000]
        for nome, rede in redes().items():
            torch.manual_seed(s); treinar_rede(rede, (X[tr], y[tr]), epocas=a.epocas, log=None)
            lf = LeitorFuzzy(rede, vistas="imagem").preparar((X[ref], y[ref]), log=lambda *x: None)
            F = lf.caracteristicas((X[te], y[te]))
            F["erro"] = F.alucinou; F["prox_p"] = F["msp"] = 1 - F.confianca; F["p"] = F.confianca
            # escores dos métodos (fora-da-dobra quando treinados), como no pipeline principal
            E = pd.DataFrame({"Softmax (MSP)": F.msp, "Deep kNN": F.vizinhos_discordam, "TTA (vistas)": F.p_vistas_std})
            g = np.zeros(len(F))
            for t1, t2 in StratifiedKFold(5, shuffle=True, random_state=s).split(F, F.erro):
                dd = F.iloc[t1]; fe = fz.selecionar(dd, [c for c in nossas if c in F], 4)
                g[t2] = fz.FuzzyAditivo().fit(dd[fe].values, dd.erro.values, fe).predict_proba(F.iloc[t2][fe].values)[:, 1]
            E["NOSSO: Fuzzy aditivo"] = g; E["erro"] = F.erro
            p = Path(a.saida) / f"fashion_{nome}" / f"semente_{s}"; p.mkdir(parents=True, exist_ok=True)
            F.to_csv(p / "features.csv.gz", index=False); E.to_csv(p / "escores.csv.gz", index=False)
            _, zres = cmp.zonas(F, False, None, s); zres.to_csv(p / "zonas.csv", index=False)
            e = F.erro.values
            L.append({"semente": s, "rede": nome, "acurácia": 1 - e.mean(), "erros": int(e.sum()),
                      "AUROC fuzzy": roc_auc_score(e, g), "AUROC softmax": roc_auc_score(e, F.msp),
                      "AUROC Deep kNN": roc_auc_score(e, F.vizinhos_discordam)})
            print(f"  semente {s} · {nome}: acurácia {1 - e.mean():.3f}, {e.sum()} erros | AUROC fuzzy {L[-1]['AUROC fuzzy']:.3f}"
                  f" × softmax {L[-1]['AUROC softmax']:.3f}", flush=True)
    pd.DataFrame(L).to_csv(Path(a.saida) / "resumo_geral.csv", index=False)


if __name__ == "__main__":
    main()
