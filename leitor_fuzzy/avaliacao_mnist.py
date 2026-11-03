"""
avaliacao_mnist.py — avaliação mais séria da biblioteca no MNIST real (5.000 dígitos do mlxtend):
5 sementes, intervalo de confiança por bootstrap e baselines mais fortes que o softmax.

    python -m leitor_fuzzy.avaliacao_mnist

Métodos (maior = mais chance de alucinação):
  Softmax (MSP)          1 − confiança da rede
  Só P                   regressão logística em (p, p²), treinada na calibração
  Deep kNN               fração dos 15 vizinhos do treino com outra resposta
  LR p + vizinhos        regressão logística em p e nas duas características de vizinhos (forte, não fuzzy)
  Fuzzy completo         o LeitorFuzzy como vem (11 candidatas, p + 3)
  Fuzzy sem vizinhos     mesmas regras, sem as duas de vizinhos: só caminho pelos blocos + vistas
  Fuzzy só caminho       p + características do caminho pelos blocos
"""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from leitor_fuzzy import LeitorFuzzy, treinar_rede
from leitor_fuzzy.exemplo_mnist import redes
from leitor_fuzzy.fuzzy import FuzzyAditivo, selecionar
from leitor_fuzzy.leitor import CARACTERISTICAS

VIZ = ["vizinhos_discordam", "dist_vizinhos"]
CAMINHO = ["p", "trocas_opiniao", "prof_decisao", "entropia_probes", "conf_fim", "salto_final"]


def _fuzzy(Fc, Ft, cand):
    y = Fc.alucinou.values
    esc = selecionar(Fc, y, [c for c in cand if c in Fc], 4)
    return FuzzyAditivo().fit(Fc[esc].values, y, esc).risco(Ft[esc].values), esc


def _lr(Fc, Ft, cols):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Fc[cols].values, Fc.alucinou.values)
    return m.predict_proba(Ft[cols].values)[:, 1]


def uma_rodada(X, y, semente, nome_rede, epocas=4):
    i = np.random.default_rng(semente).permutation(len(X))
    tr, cal, te = i[:2500], i[2500:3500], i[3500:]
    torch.manual_seed(semente)
    rede = treinar_rede(redes()[nome_rede], (X[tr], y[tr]), epocas=epocas, log=None)
    lf = LeitorFuzzy(rede, vistas="deslocamento").fit((X[tr], y[tr]), (X[cal], y[cal]), log=lambda *a: None)
    Fc, Ft = lf.caracteristicas((X[cal], y[cal])), lf.caracteristicas((X[te], y[te]))
    for F in (Fc, Ft):
        F["p2"] = F.confianca ** 2
    s = {"Softmax (MSP)": 1 - Ft.confianca.values,
         "Só P": _lr(Fc, Ft, ["confianca", "p2"]),
         "Deep kNN": Ft.vizinhos_discordam.values,
         "LR p + vizinhos": _lr(Fc, Ft, ["confianca"] + VIZ),
         "Fuzzy completo": lf.risco(None, Ft)}
    s["Fuzzy sem vizinhos"], esc_sv = _fuzzy(Fc, Ft, [c for c in CARACTERISTICAS if c not in VIZ])
    s["Fuzzy só caminho"], _ = _fuzzy(Fc, Ft, CAMINHO)
    return Ft.alucinou.values, s, lf.escolhidas, esc_sv, 1 - Ft.alucinou.mean()


def bootstrap_dif(e, a, b, n=1000, semente=0):
    r = np.random.default_rng(semente); d = []
    for _ in range(n):
        j = r.integers(0, len(e), len(e))
        if e[j].min() != e[j].max():
            d.append(roc_auc_score(e[j], a[j]) - roc_auc_score(e[j], b[j]))
    return np.percentile(d, [2.5, 97.5])


def main(sementes=(0, 1, 2, 3, 4)):
    from mlxtend.data import mnist_data
    X, y = mnist_data()
    X = torch.tensor(X / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28); y = torch.tensor(y)
    lin, ics, esc = [], [], []
    for nome in ("CNN", "MLP"):
        for sem in sementes:
            e, s, escolhidas, esc_sv, acc = uma_rodada(X, y, sem, nome)
            print(f"{nome} semente {sem}: acurácia {acc:.3f}, {e.sum()} alucinações | escolhidas: {', '.join(escolhidas)}", flush=True)
            for m, v in s.items():
                lin.append({"rede": nome, "semente": sem, "método": m, "AUROC": roc_auc_score(e, v)})
            for rival in ("Só P", "Deep kNN", "LR p + vizinhos"):
                lo, hi = bootstrap_dif(e, s["Fuzzy completo"], s[rival], semente=sem)
                ics.append({"rede": nome, "semente": sem, "contra": rival, "inf": lo, "sup": hi})
            esc.append({"rede": nome, "semente": sem, "completo": ", ".join(escolhidas), "sem vizinhos": ", ".join(esc_sv)})
    R = pd.DataFrame(lin); I = pd.DataFrame(ics)
    tab = R.groupby(["rede", "método"]).AUROC.agg(["mean", "std"]).round(3).unstack(0)
    print("\nAUROC média (desvio) em", len(sementes), "sementes:\n", tab.to_string())
    I["veredito"] = np.where(I.inf > 0, "melhor", np.where(I.sup < 0, "pior", "empate"))
    print("\nFuzzy completo contra cada rival (IC 95% bootstrap, por semente):")
    print(I.groupby(["rede", "contra"]).veredito.value_counts().unstack(fill_value=0).to_string())
    return R, I, pd.DataFrame(esc)


if __name__ == "__main__":
    main()
