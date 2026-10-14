"""
fuzzy.py — os NOSSOS dois sistemas fuzzy (regras aprendidas automaticamente) e o diagnóstico das features.

  WangMendel     Wang & Mendel (1992): cada imagem propõe uma regra com 3 condições (P + 2 features);
                 conclusão ERRO/ACERTO e fator de certeza (CF) vêm de todas as imagens compatíveis.
  FuzzyAditivo   regras de UMA condição com pesos (log-odds), somadas: "SE salto_final é ALTA → +1,3".

Cada feature tem 3 termos triangulares (BAIXA, MÉDIA, ALTA) com centros nos quantis 5/50/95%,
exceto P no binário, que tem centros fixos 0 / limiar / 1 (BAIXA = benigna com certeza, MÉDIA = no limiar,
ALTA = maligna com certeza).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-12
TERMOS = ["BAIXA", "MÉDIA", "ALTA"]
RESPOSTA = "p"


class Particoes:
    def __init__(self, centros_fixos=None):
        self.fixos = centros_fixos or {}

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


class WangMendel:
    def __init__(self, centros_fixos=None, min_suporte=0.01, min_cf=0.1):
        self.fixos, self.min_suporte, self.min_cf = centros_fixos, min_suporte, min_cf

    def fit(self, X, y, nomes):
        y = np.asarray(y).astype(int); self.nomes = list(nomes)
        self.part = Particoes(self.fixos).fit(X, nomes); M = self.part.pertinencias(X); n = len(y)
        w = np.where(y == 1, n / (2 * max(y.sum(), 1)), n / (2 * max((1 - y).sum(), 1)))      # classes balanceadas
        rot = np.stack([m.argmax(1) for m in M], 1)
        self.regras_ = []
        for ant in {tuple(r) for r in rot}:
            comp = np.prod([M[j][:, k] for j, k in enumerate(ant)], axis=0)
            sup = comp.sum() / n; b = np.array([(comp * w)[y == 0].sum(), (comp * w)[y == 1].sum()])
            if sup < self.min_suporte or b.sum() <= EPS:
                continue
            c = int(b.argmax()); cf = (b[c] - b[1 - c]) / b.sum()
            if cf >= self.min_cf:
                self.regras_.append({"ant": ant, "cons": c, "cf": float(cf), "suporte": float(sup)})
        self.regras_.sort(key=lambda r: -r["suporte"] * r["cf"])
        return self

    def predict_proba(self, X):
        M = self.part.pertinencias(X); S = np.zeros((len(M[0]), 2))
        for r in self.regras_:
            S[:, r["cons"]] += np.prod([M[j][:, k] for j, k in enumerate(r["ant"])], axis=0) * r["cf"]
        tot = S.sum(1, keepdims=True)
        return np.where(tot > EPS, S / np.maximum(tot, EPS), 0.5)

    def assinaturas(self):
        return {(r["ant"], r["cons"]) for r in self.regras_}

    def regras(self):
        return pd.DataFrame([{"regra": "SE " + " E ".join(f"{self.nomes[j]} é {TERMOS[k]}" for j, k in enumerate(r["ant"]))
                              + (" ENTÃO ERRO" if r["cons"] else " ENTÃO ACERTO"),
                              "conclusão": "ERRO" if r["cons"] else "ACERTO",
                              "certeza (CF)": round(r["cf"], 3), "suporte": round(r["suporte"], 4)} for r in self.regras_])


class FuzzyAditivo:
    def __init__(self, centros_fixos=None, C=1.0):
        self.fixos, self.C = centros_fixos, C

    def fit(self, X, y, nomes):
        self.nomes = list(nomes); self.part = Particoes(self.fixos).fit(X, nomes)
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(np.hstack(self.part.pertinencias(X)), y)
        return self

    def predict_proba(self, X):
        return self.lr.predict_proba(np.hstack(self.part.pertinencias(X)))

    def pesos(self):
        return self.lr.coef_[0]

    def regras(self):
        w = self.lr.coef_[0]
        lin = [{"regra": f"SE {n} é {TERMOS[k]}", "peso no risco de erro": round(float(w[3 * j + k]), 3),
                "efeito": "aumenta" if w[3 * j + k] > 0.05 else ("diminui" if w[3 * j + k] < -0.05 else "neutro")}
               for j, n in enumerate(self.nomes) for k in range(3)]
        lin.append({"regra": "(base)", "peso no risco de erro": round(float(self.lr.intercept_[0]), 3), "efeito": ""})
        return pd.DataFrame(lin)


# ----------------------------------------------------------------------------- seleção e diagnóstico
def auroc_alem_de_p(tab, cand, faixas=5):
    """AUROC dentro de 5 faixas de P parecido (média ponderada): o que a feature traz de NOVO em relação a P."""
    y = tab.erro.values; bins = pd.qcut(tab[RESPOSTA].rank(method="first"), faixas, labels=False).values; out = {}
    for f in cand:
        v = tab[f].values; dentro = []
        for b in range(faixas):
            m = bins == b
            if y[m].min() != y[m].max() and np.std(v[m]) > 0:
                a = roc_auc_score(y[m], v[m]); dentro.append((max(a, 1 - a), m.sum()))
        out[f] = np.average([d for d, _ in dentro], weights=[w for _, w in dentro]) if dentro else 0.5
    return out


def selecionar(tab, cand, k, max_corr=0.85):
    """P sempre entra; as outras k-1 são as de maior AUROC além de P, sem redundância (|ρ| < max_corr)."""
    cand = [c for c in cand if c != RESPOSTA and tab[c].nunique() > 1]
    alem = auroc_alem_de_p(tab, cand); esc = [RESPOSTA]
    for f in sorted(cand, key=lambda f: -alem[f]):
        if len(esc) == k:
            break
        if all(abs(spearmanr(tab[f], tab[g]).correlation) < max_corr for g in esc[1:]):
            esc.append(f)
    return esc


def estabilidade_wm(tab, feats, fixos, n=10, semente=42):
    """Jaccard médio das regras do Wang-Mendel entre reamostragens (1 = sempre as mesmas regras)."""
    r = np.random.default_rng(semente); A = []
    for _ in range(n):
        i = r.integers(0, len(tab), len(tab))
        if tab.erro.values[i].min() == tab.erro.values[i].max():
            continue
        A.append(WangMendel(fixos).fit(tab.iloc[i][feats].values, tab.erro.values[i], feats).assinaturas())
    v = [len(a & b) / max(len(a | b), 1) for j, a in enumerate(A) for b in A[j + 1:]]
    return float(np.mean(v)) if v else float("nan")


def estabilidade_aditivo(tab, feats, fixos, n=10, semente=42):
    """Correlação média (Spearman) dos pesos das regras entre reamostragens (1 = mesmos pesos)."""
    r = np.random.default_rng(semente); W = []
    for _ in range(n):
        i = r.integers(0, len(tab), len(tab))
        if tab.erro.values[i].min() == tab.erro.values[i].max():
            continue
        W.append(FuzzyAditivo(fixos).fit(tab.iloc[i][feats].values, tab.erro.values[i], feats).pesos())
    v = [spearmanr(a, b).correlation for j, a in enumerate(W) for b in W[j + 1:]]
    return float(np.nanmean(v)) if v else float("nan")


def diagnostico(tab, cand, escolhas=None):
    """Uma linha por feature: AUROC, AUROC além de P, direção, correlação com P, quantas vezes foi escolhida."""
    y = tab.erro.values; alem = auroc_alem_de_p(tab, cand); lin = []
    R = tab[cand].corr("spearman").abs().fillna(0); R = R.mask(np.eye(len(R), dtype=bool), 0)
    for f in cand:
        v = tab[f].values
        a = roc_auc_score(y, v) if np.std(v) > 0 else 0.5
        lin.append({"feature": f, "AUROC": max(a, 1 - a), "direção": "maior = erro" if a >= .5 else "menor = erro",
                    "AUROC além de P": alem.get(f, np.nan), "corr. com P": spearmanr(v, tab[RESPOSTA]).correlation if np.std(v) > 0 else 0,
                    "escolhida (folds)": sum(f in e for e in escolhas) if escolhas else np.nan,
                    "mediana acertos": np.median(v[y == 0]), "mediana erros": np.median(v[y == 1]),
                    "mais parecida com": f"{R[f].idxmax()} ({R[f].max():.2f})"})
    return pd.DataFrame(lin).sort_values("AUROC além de P", ascending=False).reset_index(drop=True)
