"""
analise_pares.py — o fuzzy aditivo com regras de DUAS condições, sempre com o p numa delas:
    "SE p é ALTA E trocas_opiniao é ALTA ENTÃO +risco"
além das regras de uma condição de hoje. Mesmas características (p + 3), mesmas dobras e mesmos dados do
pipeline principal; roda em cima do que o rodar.py já salvou (sem GPU).

    python analise_pares.py --pasta resultados --relatorio resultados/relatorio.xlsx

O relatório só é usado para pegar o limiar de cada rede no HAM10000 (termos de p fixos em 0 / limiar / 1).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config as C
import fuzzy as fz
from leitura import NOSSAS


class FuzzyPares:
    """Regras de uma condição (3 por característica) + regras 'p é A E x_j é B' (9 por característica além de p).
    A força de uma regra de duas condições é o produto das pertinências (t-norma produto)."""
    def __init__(self, fixos=None, C=1.0):
        self.fixos, self.C = fixos, C

    def _X(self, X):
        M = self.part.pertinencias(X)                       # lista de (n, 3); M[0] é o p
        cols = M + [(M[0][:, a] * M[j][:, b])[:, None] for j in range(1, len(M)) for a in range(3) for b in range(3)]
        return np.hstack(cols)

    def fit(self, X, y, nomes):
        assert nomes[0] == "p"
        self.nomes = list(nomes); self.part = fz.Particoes(self.fixos).fit(X, nomes)
        self.lr = LogisticRegression(C=self.C, max_iter=5000).fit(self._X(X), y)
        return self

    def predict_proba(self, X):
        return self.lr.predict_proba(self._X(X))

    def regras(self, top=8):
        w = self.lr.coef_[0]; T = fz.TERMOS; nome = []
        for j, n in enumerate(self.nomes):
            nome += [f"SE {n} é {T[k]}" for k in range(3)]
        for j in range(1, len(self.nomes)):
            nome += [f"SE p é {T[a]} E {self.nomes[j]} é {T[b]}" for a in range(3) for b in range(3)]
        r = pd.DataFrame({"regra": nome, "peso no risco de erro": w})
        return r.reindex(r["peso no risco de erro"].abs().sort_values(ascending=False).index).head(top)


def ic(y, a, b, n=1000, semente=0):
    r = np.random.default_rng(semente); d = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default="resultados")
    ap.add_argument("--relatorio", default=None)
    a = ap.parse_args()
    pasta = Path(a.pasta)
    lim = {}
    if a.relatorio:
        rd = pd.read_excel(a.relatorio, "redes")
        lim = {(r.experimento, int(r.semente)): r.limiar for r in rd.itertuples() if pd.notna(r.limiar)}
    L, R = [], []
    for p in sorted(pasta.glob("*/semente_*/features.csv.gz")):
        exp, sem = p.parent.parent.name, int(p.parent.name.split("_")[1])
        F = pd.read_csv(p); E = pd.read_csv(p.parent / "escores.csv.gz"); y = F.erro.values
        binario = (exp, sem) in lim
        fixos = {"p": [0.0, lim[(exp, sem)], 1.0]} if binario else None
        cand = [c for c in NOSSAS if c in F]
        s1, s2 = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=sem).split(F, y):
            d = F.iloc[tr]; feats = fz.selecionar(d, cand, C.FEATURES_ADITIVO)
            s1[te] = fz.FuzzyAditivo(fixos).fit(d[feats].values, y[tr], feats).predict_proba(F.iloc[te][feats].values)[:, 1]
            s2[te] = FuzzyPares(fixos).fit(d[feats].values, y[tr], feats).predict_proba(F.iloc[te][feats].values)[:, 1]
        lo, hi = ic(y, s2, s1, semente=sem)
        L.append({"experimento": exp, "semente": sem, "uma condição": roc_auc_score(y, s1), "duas condições (p E x)": roc_auc_score(y, s2),
                  "Só P": roc_auc_score(y, E["Só P"]), "Chen et al. 2019": roc_auc_score(y, E["Chen et al. 2019"]),
                  "ganho": roc_auc_score(y, s2) - roc_auc_score(y, s1), "IC inf": lo, "IC sup": hi})
        feats = fz.selecionar(F, cand, C.FEATURES_ADITIVO)
        R.append(FuzzyPares(fixos).fit(F[feats].values, y, feats).regras().assign(experimento=exp, semente=sem))
        print(f"  {exp} semente {sem}: uma {L[-1]['uma condição']:.3f} | duas {L[-1]['duas condições (p E x)']:.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]", flush=True)
    L = pd.DataFrame(L); L["veredito"] = np.where(L["IC inf"] > 0, "melhor", np.where(L["IC sup"] < 0, "pior", "empate"))
    pd.set_option("display.width", 220)
    print("\nAUROC (média das sementes):")
    print(L.groupby("experimento")[["Só P", "Chen et al. 2019", "uma condição", "duas condições (p E x)", "ganho"]].mean().round(3).to_string())
    print("\nDuas condições × uma condição, por rodada:", L.veredito.value_counts().to_dict())
    arq = pasta / "pares.xlsx"
    with pd.ExcelWriter(arq) as xw:
        L.round(4).to_excel(xw, sheet_name="auroc", index=False)
        pd.concat(R).round(3).to_excel(xw, sheet_name="regras_mais_fortes", index=False)
    print("Planilha:", arq)


if __name__ == "__main__":
    main()
