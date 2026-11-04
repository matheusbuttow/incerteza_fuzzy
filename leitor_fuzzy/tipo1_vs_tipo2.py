"""
tipo1_vs_tipo2.py — fuzzy aditivo tipo 1 x tipo 2 no MNIST real, com as MESMAS redes, sementes, características e metas.
Regras aprendidas em duas redes e aplicadas na terceira. Zonas com limiares com garantia (Clopper–Pearson 95%),
escolhidos nas redes de origem:
  tipo 1: nota s;            seguro: s < t_s · alarme: s > t_a · incerto: entre os dois
  tipo 2: intervalo [s_inf, s_sup]; seguro: s_sup < t_s · alarme: s_inf > t_a · incerto: o resto

    python -m leitor_fuzzy.tipo1_vs_tipo2
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimentos" / "ham_tipo2"))
import fuzzy_t2 as t2  # noqa: E402

from leitor_fuzzy import LeitorFuzzy, treinar_rede  # noqa: E402
from leitor_fuzzy.leitor import CARACTERISTICAS  # noqa: E402
from leitor_fuzzy.tipo2_melhorias import META_SEGURO, VISTAS, limiar_alarme, limiar_seguro  # noqa: E402
from leitor_fuzzy.tipo2_mnist import redes  # noqa: E402


def zonas(y, lo, hi, ts, ta):
    seg = hi < ts; al = (lo > ta) & ~seg; inc = ~seg & ~al
    return {"seguro %": seg.mean(), "erro no seguro": y[seg].mean() if seg.any() else np.nan,
            "alarme %": al.mean(), "erro no alarme": y[al].mean() if al.any() else np.nan,
            "incerto %": inc.mean(), "erro no incerto": y[inc].mean() if inc.any() else np.nan,
            "erros pegos (alarme)": y[al].sum() / y.sum(), "erros pegos (alarme+incerto)": y[~seg].sum() / y.sum(),
            "erros que escapam": y[seg].sum() / y.sum(), "erros na rede": int(y.sum()),
            "escaparam (n)": int(y[seg].sum()), "alarmes falsos (n)": int((~y[al].astype(bool)).sum())}


def comparar(tabs, cand, k=4):
    lin = []
    for alvo in tabs:
        fontes = [s for s in tabs if s != alvo]
        src = pd.concat([tabs[s].assign(_g=i) for i, s in enumerate(fontes)], ignore_index=True)
        ys, gs = src.erro.values, src._g.values
        feats = t2.selecionar(src, cand, k, ("prox_p",))
        Xs, T = src[feats].values, tabs[alvo]; Xa, y = T[feats].values, T.erro.values
        # tipo 1: um conjunto de termos (picos nas duas redes de origem juntas)
        C1 = t2.centros(Xs)
        m1 = t2.FuzzyAditivo().fit(t2.pert_t1(Xs, C1), ys, feats, gs)
        s_src, s_alvo = m1.nota_t1(t2.pert_t1(Xs, C1)), m1.nota_t1(t2.pert_t1(Xa, C1))
        z1 = zonas(y, s_alvo, s_alvo, limiar_seguro(s_src, ys, True), limiar_alarme(s_src, ys, True))
        lin.append({"alvo": alvo, "tipo": "tipo 1", "AUROC": roc_auc_score(y, s_alvo), **z1})
        # tipo 2: envelope dos termos de cada rede de origem
        lista_C = [t2.centros(src[src._g == i][feats].values) for i in range(len(fontes))]
        Mi, Ms = t2.pert_t2(Xs, lista_C)
        m2 = t2.FuzzyAditivo().fit((Mi + Ms) / 2, ys, feats, gs)
        si_s, ss_s, _ = m2.nota_t2(Mi, Ms)
        si, ss, sm = m2.nota_t2(*t2.pert_t2(Xa, lista_C))
        z2 = zonas(y, si, ss, limiar_seguro(ss_s, ys, True), limiar_alarme(si_s, ys, True))
        lin.append({"alvo": alvo, "tipo": "tipo 2", "AUROC": roc_auc_score(y, sm), "largura média": (ss - si).mean(), **z2})
    return lin


def main(sementes=(3, 4, 5, 6, 7), epocas=4):
    from mlxtend.data import mnist_data
    X, y = mnist_data()
    X = torch.tensor(X / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28); y = torch.tensor(y)
    cand = [c for c in CARACTERISTICAS if c != "p"] + ["prox_p"]
    R = []
    for s in sementes:
        i = np.random.default_rng(s).permutation(len(X)); tr, te = i[:2500], i[2500:]; tabs = {}
        for nome, rede in redes().items():
            torch.manual_seed(s); treinar_rede(rede, (X[tr], y[tr]), epocas=epocas, log=None)
            lf = LeitorFuzzy(rede, vistas=VISTAS["giro 10°"]).preparar((X[tr], y[tr]), log=lambda *a: None)
            F = lf.caracteristicas((X[te], y[te])); F["erro"], F["prox_p"] = F.alucinou, 1 - F.confianca
            tabs[nome] = F
        R += [{"semente": s, **d} for d in comparar(tabs, cand)]
        print(f"  semente {s} pronta", flush=True)
    R = pd.DataFrame(R)
    pd.set_option("display.width", 250)
    cols = ["AUROC", "seguro %", "erro no seguro", "alarme %", "erro no alarme", "incerto %", "erro no incerto",
            "erros pegos (alarme)", "erros pegos (alarme+incerto)", "erros que escapam"]
    print(f"\nMédia (3 redes alvo × {len(sementes)} sementes), meta: erro no seguro ≤ {META_SEGURO:.0%}")
    print(R.groupby("tipo")[cols].mean().round(3).T.to_string())
    print("\nEm números (média por rede, 2.500 imagens):")
    print(R.groupby("tipo")[["erros na rede", "escaparam (n)", "alarmes falsos (n)"]].mean().round(0).to_string())
    print(f"\nVezes em que o erro no seguro passou de {META_SEGURO:.0%} (de {len(R) // 2}):")
    print(R.assign(passou=R["erro no seguro"] > META_SEGURO).groupby("tipo").passou.sum().to_string())
    P = R.pivot_table(index=["semente", "alvo"], columns="tipo", values=["erros que escapam", "seguro %", "AUROC"])
    for c, melhor in [("erros que escapam", "menor"), ("seguro %", "maior"), ("AUROC", "maior")]:
        d = P[c]["tipo 2"] - P[c]["tipo 1"]
        ganha = (d < 0).sum() if melhor == "menor" else (d > 0).sum()
        print(f"  {c}: tipo 2 − tipo 1 = {d.mean():+.4f} em média; tipo 2 melhor em {ganha} de {len(d)}")
    print("\nPor rede alvo:")
    print(R.groupby(["alvo", "tipo"])[["AUROC", "seguro %", "erro no seguro", "incerto %", "erros que escapam"]].mean().round(3).to_string())
    print("\nLargura média do intervalo do tipo 2:", round(R["largura média"].mean(), 4))
    return R


if __name__ == "__main__":
    R = main()
    R.to_csv("tipo1_vs_tipo2.csv", index=False)
