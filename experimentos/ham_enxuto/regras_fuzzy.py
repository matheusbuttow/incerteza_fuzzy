"""
regras_fuzzy.py — árvore de decisão fuzzy que aprende, sozinha, regras SE...ENTÃO ERRO/ACERTO,
e as ferramentas para avaliar e diagnosticar as features.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
TERMOS = ["BAIXA", "MÉDIA", "ALTA"]


# ----------------------------------------------------------------------------- partições
class Particoes:
    """3 termos triangulares por feature, centros nos quantis 5%, 50%, 95% (somam 1)."""

    def fit(self, X):
        C = np.nanquantile(np.asarray(X, float), [0.05, 0.5, 0.95], axis=0).T
        for j in range(len(C)):
            for k in (1, 2):
                C[j, k] = max(C[j, k], C[j, k - 1] + 1e-6 * (abs(C[j, k - 1]) + 1))
        self.centros = C
        return self

    def pertinencias(self, X):
        X = np.asarray(X, float); out = []
        for j, c in enumerate(self.centros):
            x = np.nan_to_num(X[:, j], nan=c[1])
            M = np.zeros((len(x), 3))
            M[:, 0] = x <= c[0]; M[:, 2] = x >= c[2]
            for k in (0, 1):
                d = (x >= c[k]) & (x < c[k + 1]); t = (x[d] - c[k]) / (c[k + 1] - c[k])
                M[d, k] = 1 - t; M[d, k + 1] = t
            out.append(M)
        return out


# ----------------------------------------------------------------------------- árvore fuzzy
class ArvoreFuzzy:
    """Fuzzy ID3: em cada nó, divide nos termos BAIXA/MÉDIA/ALTA da feature com maior ganho de informação."""

    def __init__(self, prof_max=2, min_peso=0.03, pureza=0.95, balancear=True):
        self.prof_max, self.min_peso, self.pureza, self.balancear = prof_max, min_peso, pureza, balancear

    def _H(self, w):
        b = np.array([(w * (self._y == c)).sum() * self._cw[c] for c in (0, 1)])
        if b.sum() <= EPS:
            return 0.0, np.array([.5, .5])
        p = b / b.sum()
        return float(-(p * np.log2(p + EPS)).sum()), p

    def fit(self, X, y, nomes):
        self.nomes = list(nomes); y = np.asarray(y).astype(int)
        self.part = Particoes().fit(X)
        self._M, self._y, self._n = self.part.pertinencias(X), y, len(y)
        self._cw = {c: (len(y) / (2 * max((y == c).sum(), 1)) if self.balancear else 1.0) for c in (0, 1)}
        self.raiz = self._crescer(np.ones(len(y)), (), 0, None)
        del self._M, self._y
        return self

    def _crescer(self, w, usados, prof, p_pai):
        H, p = self._H(w); peso = w.sum() / self._n
        if peso < self.min_peso:
            return {"folha": True, "p": p if p_pai is None else p_pai, "peso": peso}
        folha = {"folha": True, "p": p, "peso": peso}
        if prof >= self.prof_max or p.max() >= self.pureza:
            return folha
        melhor, ganho_max = None, 1e-3
        for j in range(len(self.nomes)):
            if j in usados: continue
            filhos = [w * self._M[j][:, k] for k in range(3)]
            tot = sum(f.sum() for f in filhos) + EPS
            g = H - sum(f.sum() / tot * self._H(f)[0] for f in filhos)
            if g > ganho_max: melhor, ganho_max = (j, filhos), g
        if melhor is None:
            return folha
        j, filhos = melhor
        return {"folha": False, "feat": j, "peso": peso, "p": p,
                "filhos": [self._crescer(f, usados + (j,), prof + 1, p) for f in filhos]}

    def predict_proba(self, X):
        M = self.part.pertinencias(X); n = len(M[0]); num = np.zeros(n); den = np.zeros(n)

        def desce(no, g):
            if no["folha"]:
                num[:] += g * no["p"][1]; den[:] += g; return
            for k, filho in enumerate(no["filhos"]):
                gk = g * M[no["feat"]][:, k]
                if gk.max() > 0: desce(filho, gk)
        desce(self.raiz, np.ones(n))
        p1 = np.where(den > EPS, num / np.maximum(den, EPS), .5)
        return np.c_[1 - p1, p1]

    def _folhas(self):
        out = []
        def rec(no, cam):
            if no["folha"]: out.append((cam, no)); return
            for k, f in enumerate(no["filhos"]): rec(f, cam + [(no["feat"], k)])
        rec(self.raiz, []); return out

    @property
    def n_regras(self):
        return len(self._folhas())

    def assinaturas(self):
        return {(tuple(sorted(c)), int(np.argmax(no["p"]))) for c, no in self._folhas()}

    def regras(self):
        lin = []
        for cam, no in sorted(self._folhas(), key=lambda t: -t[1]["peso"]):
            se = " E ".join(f"{self.nomes[j]} é {TERMOS[k]}" for j, k in cam) or "(sempre)"
            c = int(np.argmax(no["p"]))
            lin.append({"regra": f"SE {se} ENTÃO {'ERRO' if c else 'ACERTO'}",
                        "confiança": round(float(no["p"][c]), 3), "suporte": round(float(no["peso"]), 4)})
        return pd.DataFrame(lin)


# ----------------------------------------------------------------------------- fuzzy aditivo
class FuzzyAditivo:
    """Regras de UMA condição cada, somadas: 'SE salto_final é ALTA → +1,28 no risco de erro'.
    Os pesos (em log-odds) são aprendidos por regressão logística sobre os graus de pertinência.
    Peso positivo = aumenta o risco de a rede errar; negativo = diminui."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y, nomes):
        self.nomes = list(nomes); self.part = Particoes().fit(X)
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(np.hstack(self.part.pertinencias(X)), y)
        return self

    def predict_proba(self, X):
        return self.lr.predict_proba(np.hstack(self.part.pertinencias(X)))

    @property
    def n_regras(self):
        return 3 * len(self.nomes)

    def regras(self):
        w = self.lr.coef_[0]
        lin = [{"regra": f"SE {n} é {TERMOS[k]}", "peso no risco de erro": round(float(w[3 * j + k]), 3),
                "efeito": "aumenta" if w[3 * j + k] > 0.05 else ("diminui" if w[3 * j + k] < -0.05 else "neutro")}
               for j, n in enumerate(self.nomes) for k in range(3)]
        lin.append({"regra": "(base)", "peso no risco de erro": round(float(self.lr.intercept_[0]), 3), "efeito": ""})
        return pd.DataFrame(lin)



# ----------------------------------------------------------------------------- Wang-Mendel
class WangMendel:
    """Wang & Mendel (1992): cada exemplo propõe a regra com o termo de maior grau em cada feature;
    a conclusão (ERRO/ACERTO) e o fator de certeza (CF) vêm de todos os exemplos compatíveis, com classes balanceadas.
    Na inferência, todas as regras votam, pesadas por compatibilidade × CF."""

    def __init__(self, min_suporte=0.01, min_cf=0.1):
        self.min_suporte, self.min_cf = min_suporte, min_cf

    def fit(self, X, y, nomes):
        y = np.asarray(y).astype(int); self.nomes = list(nomes); self.part = Particoes().fit(X)
        M = self.part.pertinencias(X); n = len(y)
        w = np.where(y == 1, n / (2 * max(y.sum(), 1)), n / (2 * max((1 - y).sum(), 1)))
        antecedentes = {tuple(int(np.argmax(M[j][i])) for j in range(len(M))) for i in range(n)}
        self.regras_ = []
        for ant in antecedentes:
            comp = np.prod([M[j][:, k] for j, k in enumerate(ant)], axis=0)
            sup = comp.sum() / n
            b = np.array([(comp * w)[y == 0].sum(), (comp * w)[y == 1].sum()])
            if sup < self.min_suporte or b.sum() <= EPS: continue
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

    @property
    def n_regras(self):
        return len(self.regras_)

    def assinaturas(self):
        return {(r["ant"], r["cons"]) for r in self.regras_}

    def regras(self):
        return pd.DataFrame([{"regra": "SE " + " E ".join(f"{self.nomes[j]} é {TERMOS[k]}" for j, k in enumerate(r["ant"]))
                              + (" ENTÃO ERRO" if r["cons"] else " ENTÃO ACERTO"),
                              "certeza (CF)": round(r["cf"], 3), "suporte": round(r["suporte"], 3)} for r in self.regras_])

# ----------------------------------------------------------------------------- seleção e avaliação
def selecionar(tab, cand, k=4, max_corr=0.85):
    """Top-k por AUROC (para detectar o erro), sem features muito correlacionadas entre si."""
    rk = sorted(cand, key=lambda f: -abs(roc_auc_score(tab.erro, tab[f]) - 0.5) if tab[f].nunique() > 1 else 0)
    esc = []
    for f in rk:
        if all(abs(spearmanr(tab[f], tab[g]).correlation) < max_corr for g in esc):
            esc.append(f)
        if len(esc) == k: break
    return esc


def comparar(tab, cand, k=4, prof=2, n_folds=5, seed=42):
    """Validação cruzada: só P × regressão logística × árvore fuzzy. Seleção de features dentro de cada fold."""
    y = tab.erro.values
    oof = {m: np.zeros(len(y)) for m in ("Só P", "Regressão logística", "Árvore fuzzy", "Fuzzy aditivo")}
    escolhas, n_regras = [], []
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(tab, y):
        d = tab.iloc[tr]; feats = selecionar(d, cand, k); escolhas.append(feats)
        oof["Só P"][te] = tab.prox_p.values[te]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(d[feats], y[tr])
        oof["Regressão logística"][te] = lr.predict_proba(tab.iloc[te][feats])[:, 1]
        arv = ArvoreFuzzy(prof).fit(d[feats].values, y[tr], feats)
        oof["Árvore fuzzy"][te] = arv.predict_proba(tab.iloc[te][feats].values)[:, 1]; n_regras.append(arv.n_regras)
        oof["Fuzzy aditivo"][te] = FuzzyAditivo().fit(d[feats].values, y[tr], feats).predict_proba(tab.iloc[te][feats].values)[:, 1]
    r = np.random.default_rng(0); linhas = []
    for m, s in oof.items():
        d = [roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], oof["Só P"][i])
             for i in (r.integers(0, len(y), len(y)) for _ in range(1000)) if y[i].min() != y[i].max()]
        o = np.argsort(s); manter = o[: int(round(.8 * len(y)))]
        linhas.append({"método": m, "AUROC": roc_auc_score(y, s), "AUPRC": average_precision_score(y, s),
                       "ganho sobre só P": roc_auc_score(y, s) - roc_auc_score(y, oof["Só P"]),
                       "IC 95% do ganho": "—" if m == "Só P" else f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]",
                       "erro com 20% encaminhados": y[manter].mean(), "erro sem encaminhar": y.mean(),
                       "nº regras": np.mean(n_regras) if m == "Árvore fuzzy" else (3 * k if m == "Fuzzy aditivo" else np.nan)})
    return pd.DataFrame(linhas), escolhas, oof


def estabilidade(tab, feats, prof=2, n=10, seed=42):
    """Jaccard médio das regras entre reamostragens bootstrap (1 = sempre as mesmas)."""
    r = np.random.default_rng(seed); A = []
    for _ in range(n):
        i = r.integers(0, len(tab), len(tab))
        A.append(ArvoreFuzzy(prof).fit(tab.iloc[i][feats].values, tab.erro.values[i], feats).assinaturas())
    return float(np.mean([len(a & b) / max(len(a | b), 1) for j, a in enumerate(A) for b in A[j + 1:]]))


def diagnostico(tab, cand, escolhas=None, faixas=5):
    """Uma linha por feature, para decidir o que melhorar:
       AUROC           o quanto ela sozinha separa erros de acertos (0,5 = nada)
       AUROC além de P o mesmo, mas dentro de faixas de P parecido: o que ela traz de NOVO em relação a P
       corr. com P     redundância com P (perto de ±1 = só repete P)
       escolhida       em quantos folds a seleção a escolheu
       mediana acerto / erro, nº de valores distintos (poucos = feature grosseira, pouca informação)
       mais parecida com: a outra feature mais correlacionada (perto de 1 = redundante)"""
    y = tab.erro.values
    bins = pd.qcut(tab.prox_p.rank(method="first"), faixas, labels=False)
    lin = []
    for f in cand:
        v = tab[f].values
        if np.nanstd(v) == 0:
            lin.append({"feature": f, "AUROC": .5, "AUROC além de P": .5, "corr. com P": 0}); continue
        a = roc_auc_score(y, v); dentro = []
        for b in range(faixas):
            m = bins.values == b
            if y[m].min() != y[m].max() and np.std(v[m]) > 0:
                ab = roc_auc_score(y[m], v[m]); dentro.append((max(ab, 1 - ab), m.sum()))
        lin.append({"feature": f, "AUROC": max(a, 1 - a), "direção": "maior = erro" if a >= .5 else "menor = erro",
                    "AUROC além de P": np.average([d for d, _ in dentro], weights=[w for _, w in dentro]) if dentro else np.nan,
                    "corr. com P": spearmanr(v, tab.prox_p).correlation,
                    "escolhida (folds)": sum(f in e for e in escolhas) if escolhas else np.nan,
                    "mediana acertos": np.median(v[y == 0]), "mediana erros": np.median(v[y == 1]),
                    "nº valores distintos": len(np.unique(v))})
    out = pd.DataFrame(lin)
    R = tab[cand].corr("spearman").abs().fillna(0)
    R = R.mask(np.eye(len(R), dtype=bool), 0)
    out["mais parecida com"] = [f"{R[f].idxmax()} ({R[f].max():.2f})" if f in R else "" for f in out.feature]
    return out.sort_values("AUROC além de P", ascending=False).reset_index(drop=True)
