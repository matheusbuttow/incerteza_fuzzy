"""
fuzzy.py — o sistema fuzzy aditivo (TSK de ordem zero com regras de um antecedente) e a seleção de características.
Mesma lógica de ../fuzzy.py do artigo, só o necessário para a biblioteca.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TERMOS = ["BAIXA", "MÉDIA", "ALTA"]


class Particoes:
    """Três termos triangulares por característica (partição forte), picos nos quantis 5/50/95%."""
    def __init__(self, fixos=None):
        self.fixos = fixos or {}

    def fit(self, X, nomes):
        Cq = np.nanquantile(np.asarray(X, float), [0.05, 0.5, 0.95], axis=0).T
        for j, n in enumerate(nomes):
            if n in self.fixos:
                Cq[j] = self.fixos[n]
            for k in (1, 2):
                Cq[j, k] = max(Cq[j, k], Cq[j, k - 1] + 1e-6 * (abs(Cq[j, k - 1]) + 1))
        self.centros = Cq
        return self

    def pertinencias(self, X):
        X = np.asarray(X, float); out = []
        for j, c in enumerate(self.centros):
            x = np.nan_to_num(X[:, j], nan=c[1]); M = np.zeros((len(x), 3))
            M[:, 0] = x <= c[0]; M[:, 2] = x >= c[2]
            for k in (0, 1):
                d = (x >= c[k]) & (x < c[k + 1]); t = (x[d] - c[k]) / (c[k + 1] - c[k])
                M[d, k] = 1 - t; M[d, k + 1] = t
            out.append(M)
        return out


class FuzzyAditivo:
    """r(x) = σ(b + Σ_j Σ_k w_jk μ_jk(x_j)): uma regra 'SE x_j é TERMO' por par (característica, termo)."""
    def __init__(self, fixos=None, C=1.0):
        self.fixos, self.C = fixos, C

    def fit(self, X, y, nomes):
        self.nomes = list(nomes); self.part = Particoes(self.fixos).fit(X, nomes)
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(np.hstack(self.part.pertinencias(X)), y)
        return self

    def risco(self, X):
        return self.lr.predict_proba(np.hstack(self.part.pertinencias(X)))[:, 1]

    def contribuicoes(self, X):
        """(n, d, 3): w_jk μ_jk(x_j) de cada regra para cada exemplo."""
        M = np.stack(self.part.pertinencias(X), 1)
        return M * self.lr.coef_[0].reshape(len(self.nomes), 3)[None]

    def regras(self):
        w = self.lr.coef_[0]
        return [(f"SE {n} é {TERMOS[k]}", float(w[3 * j + k])) for j, n in enumerate(self.nomes) for k in range(3)]


def auroc_alem_de_p(F, y, cand, resposta="p", faixas=5):
    """AUROC dentro de faixas de p parecido: o que cada característica traz de NOVO em relação a p."""
    import pandas as pd
    bins = pd.qcut(pd.Series(F[resposta]).rank(method="first"), faixas, labels=False).values; out = {}
    for f in cand:
        v = F[f]; dentro = []
        for b in range(faixas):
            m = bins == b
            if y[m].min() != y[m].max() and np.std(v[m]) > 0:
                a = roc_auc_score(y[m], v[m]); dentro.append((max(a, 1 - a), m.sum()))
        out[f] = np.average([d for d, _ in dentro], weights=[w for _, w in dentro]) if dentro else 0.5
    return out


def selecionar(F, y, cand, k=4, max_corr=0.85, resposta="p"):
    """p sempre entra; as outras k-1 são as que mais acrescentam além de p, sem redundância (|ρ| < max_corr)."""
    cand = [c for c in cand if c != resposta and np.std(F[c]) > 0]
    alem = auroc_alem_de_p(F, y, cand, resposta); esc = [resposta]
    for f in sorted(cand, key=lambda f: -alem[f]):
        if len(esc) == k:
            break
        if all(abs(spearmanr(F[f], F[g]).correlation) < max_corr for g in esc[1:]):
            esc.append(f)
    return esc
