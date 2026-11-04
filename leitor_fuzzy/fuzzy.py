"""
fuzzy.py — o sistema fuzzy aditivo (TSK de ordem zero com regras de um antecedente), tipo 1 e tipo 2,
a seleção de características e os limiares das três zonas. Mesma lógica de ../fuzzy.py e de
../experimentos/ham_tipo2/fuzzy_t2.py, só o necessário para a biblioteca.
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
    """Tipo 1. r(x) = σ(b + Σ_j Σ_k w_jk μ_jk(x_j)): uma regra 'SE x_j é TERMO' por par (característica, termo)."""
    def __init__(self, fixos=None, C=1.0):
        self.fixos, self.C = fixos, C

    def fit(self, X, y, nomes):
        self.nomes = list(nomes); self.part = Particoes(self.fixos).fit(X, nomes)
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(np.hstack(self.part.pertinencias(X)), y)
        return self

    def risco(self, X):
        return self.lr.predict_proba(np.hstack(self.part.pertinencias(X)))[:, 1]

    def intervalo(self, X):
        s = self.risco(X)
        return s, s, s

    def contribuicoes(self, X):
        """(n, d, 3): w_jk μ_jk(x_j) de cada regra para cada exemplo."""
        M = np.stack(self.part.pertinencias(X), 1)
        return M * self.lr.coef_[0].reshape(len(self.nomes), 3)[None]

    def regras(self):
        w = self.lr.coef_[0]
        return [(f"SE {n} é {TERMOS[k]}", float(w[3 * j + k])) for j, n in enumerate(self.nomes) for k in range(3)]


class FuzzyAditivo2:
    """Tipo 2 intervalar. Cada termo tem uma FAIXA de picos em vez de um pico só: os picos (quantis 5/50/95%)
    são recalculados em `n_boot` reamostragens da calibração, e a pertinência vira o intervalo
    [mínimo, máximo] entre elas. Os pesos são aprendidos no meio da faixa. Como a nota é linear nas
    pertinências, o intervalo da saída sai exato:
        z_inf = b + Σ (w>0 ? w·μ_inf : w·μ_sup),   z_sup = b + Σ (w>0 ? w·μ_sup : w·μ_inf)."""
    def __init__(self, fixos=None, C=1.0, n_boot=10, semente=0):
        self.fixos, self.C, self.n_boot, self.semente = fixos, C, n_boot, semente

    def _envelope(self, X):
        Ms = np.stack([np.stack(p.pertinencias(X), 1) for p in self.parts])        # (B, n, d, 3)
        return Ms.min(0), Ms.max(0)

    def fit(self, X, y, nomes):
        X = np.asarray(X, float); self.nomes = list(nomes); r = np.random.default_rng(self.semente)
        self.part = Particoes(self.fixos).fit(X, nomes)                              # picos com todos os dados
        self.parts = [self.part] + [Particoes(self.fixos).fit(X[r.integers(0, len(X), len(X))], nomes)
                                    for _ in range(self.n_boot)]
        lo, hi = self._envelope(X)
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(((lo + hi) / 2).reshape(len(X), -1), y)
        self.w, self.b = self.lr.coef_[0], self.lr.intercept_[0]
        return self

    def intervalo(self, X):
        """(nota inferior, nota superior, nota pontual) de cada exemplo."""
        lo, hi = self._envelope(np.asarray(X, float)); n = len(lo)
        lo, hi = lo.reshape(n, -1), hi.reshape(n, -1); pos = self.w > 0
        sig = lambda z: 1 / (1 + np.exp(-z))
        s_i = sig(lo[:, pos] @ self.w[pos] + hi[:, ~pos] @ self.w[~pos] + self.b)
        s_s = sig(hi[:, pos] @ self.w[pos] + lo[:, ~pos] @ self.w[~pos] + self.b)
        return s_i, s_s, np.clip(sig(((lo + hi) / 2) @ self.w + self.b), s_i, s_s)

    def risco(self, X):
        return self.intervalo(X)[2]

    def contribuicoes(self, X):
        lo, hi = self._envelope(np.asarray(X, float))
        return (lo + hi) / 2 * self.w.reshape(len(self.nomes), 3)[None]

    def regras(self):
        return [(f"SE {n} é {TERMOS[k]}", float(self.w[3 * j + k])) for j, n in enumerate(self.nomes) for k in range(3)]


# ----------------------------------------------------------------------------- três zonas
def _taxa(erros, n, garantia, lado):
    """Taxa de erro observada, ou o limite de Clopper–Pearson (95%) quando garantia=True."""
    if not garantia:
        return erros / n
    from scipy.stats import beta
    return beta.ppf(0.95, erros + 1, n - erros) if lado == "sup" else beta.ppf(0.05, np.maximum(erros, 1e-9), n - erros + 1)


def limiar_seguro(s, y, meta, garantia=True):
    """Maior t tal que, entre os exemplos com nota < t, a taxa de erro (ou o seu limite superior 95%) fica ≤ meta."""
    s, y = np.asarray(s, float), np.asarray(y); o = np.argsort(s, kind="stable"); n = np.arange(1, len(y) + 1)
    ok = np.where(_taxa(np.cumsum(y[o]), n, garantia, "sup") <= meta)[0]
    return s[o][ok[-1]] + 1e-12 if len(ok) else -np.inf


def limiar_alarme(s, y, meta, garantia=True):
    """Menor t tal que, entre os exemplos com nota > t, a taxa de erro (ou o seu limite inferior 95%) fica ≥ meta."""
    s, y = np.asarray(s, float), np.asarray(y); o = np.argsort(-s, kind="stable"); n = np.arange(1, len(y) + 1)
    ok = np.where(_taxa(np.cumsum(y[o]), n, garantia, "inf") >= meta)[0]
    return s[o][ok[-1]] - 1e-12 if len(ok) else np.inf


def zona(s_inf, s_sup, t_seg, t_al):
    """'seguro' se s_sup < t_seg; 'alarme' se s_inf > t_al; senão 'incerto'. No tipo 1, s_inf = s_sup."""
    seg = np.asarray(s_sup) < t_seg; al = (np.asarray(s_inf) > t_al) & ~seg
    return np.where(seg, "seguro", np.where(al, "alarme", "incerto"))


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
