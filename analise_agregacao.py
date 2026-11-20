"""
analise_agregacao.py — o sistema fuzzy com outra AGREGAÇÃO (integral de Choquet) e outras PERTINÊNCIAS (gaussianas).
Não usa a GPU: lê as características que o rodar.py já salvou (features.csv.gz) e usa as mesmas partições e a mesma
seleção de características do sistema original.

Quatro versões, todas com p + 3 características escolhidas em cada partição:
  1. triangular + soma           o sistema atual (regras de uma condição somadas, como no artigo)
  2. gaussiana + soma            os mesmos centros (percentis 5/50/95), termos gaussianos normalizados para somar 1
  3. triangular + Choquet 2-ad.  cada característica vira um "risco parcial" em [0, 1] (os seus 3 termos triangulares);
                                 a integral de Choquet 2-aditiva junta os 4 riscos parciais, com interação entre PARES
  4. triangular + Choquet        o mesmo com a medida fuzzy completa (interações de qualquer tamanho)

A integral de Choquet usa a representação de Möbius: Ch(g) = Σ_A m(A) · min_{i∈A} g_i, com as restrições de monotonicidade
(a medida nunca diminui quando se acrescenta uma característica). Os coeficientes são aprendidos com perda logística e
penalização L2 igual à do sistema atual. Da medida saem:
  · o valor de Shapley de cada característica (a sua importância no risco);
  · o índice de interação de cada par: > 0 = complementares (o risco só sobe quando AS DUAS indicam erro, como um E);
                                       < 0 = substitutas/redundantes (basta UMA indicar erro, como um OU).

    python analise_agregacao.py                    # todas as rodadas da grade → resultados/agregacao.xlsx
    python analise_agregacao.py --exp cifar10n_worst cifar10n_worst_vgg16
    python analise_agregacao.py --pasta resultados_teste_rapido
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config as C
import fuzzy as fz
from analise_geral import delong, rodadas
from leitura import NOSSAS

VERSOES = ["triangular + soma", "gaussiana + soma", "triangular + Choquet 2-aditiva", "triangular + Choquet"]
BASE = VERSOES[0]


# ----------------------------------------------------------------------------- pertinências gaussianas
class ParticoesGaussianas(fz.Particoes):
    """Mesmos centros das triangulares. BAIXA e ALTA são ombros (valem 1 além do centro), MÉDIA é uma gaussiana.
    A largura faz cada termo valer 0,5 no meio do caminho até o vizinho (igual ao cruzamento das triangulares);
    no fim, os 3 termos são normalizados para somar 1 (como as triangulares), para os pesos continuarem legíveis."""

    def pertinencias(self, X):
        X = np.asarray(X, float); out = []
        for j, c in enumerate(self.centros):
            x = np.nan_to_num(X[:, j], nan=c[1])
            s01 = (c[1] - c[0]) / 2.3548; s12 = (c[2] - c[1]) / 2.3548                    # 2√(2 ln 2) = 2,3548
            g = lambda x, m, s: np.exp(-0.5 * ((x - m) / max(s, 1e-12)) ** 2)
            M = np.zeros((len(x), 3))
            M[:, 0] = np.where(x <= c[0], 1.0, g(x, c[0], s01))
            M[:, 1] = np.where(x <= c[1], g(x, c[1], s01), g(x, c[1], s12))
            M[:, 2] = np.where(x >= c[2], 1.0, g(x, c[2], s12))
            out.append(M / M.sum(1, keepdims=True))
        return out


class FuzzyAditivoGauss(fz.FuzzyAditivo):
    def fit(self, X, y, nomes):
        self.nomes = list(nomes); self.part = ParticoesGaussianas(self.fixos).fit(X, nomes)
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(np.hstack(self.part.pertinencias(X)), y)
        return self


# ----------------------------------------------------------------------------- integral de Choquet
class FuzzyChoquet:
    """Etapa 1: o fuzzy aditivo atual dá, para cada característica, a contribuição c_f = Σ_termos μ·peso;
    reescalada para [0, 1] (0 = o termo de menor peso, 1 = o de maior peso), vira o risco parcial g_f.
    Etapa 2: risco = sigmoide(a + Σ_A β(A) · min_{i∈A} g_i), β ≥ 0 nas restrições de monotonicidade; m = β / Σβ."""

    def __init__(self, k_aditiva=None, C=1.0):
        self.k, self.C = k_aditiva, C

    def _parciais(self, X):
        M = self.etapa1.part.pertinencias(X); w = self.etapa1.pesos().reshape(-1, 3)
        G = np.column_stack([Mj @ wj for Mj, wj in zip(M, w)])
        lo, hi = w.min(1), w.max(1)
        return np.clip((G - lo) / np.maximum(hi - lo, 1e-9), 0, 1)

    def _z(self, G):
        return np.column_stack([G[:, list(A)].min(1) for A in self.subconj])

    def fit(self, X, y, nomes):
        self.nomes = list(nomes); n = len(nomes)
        self.etapa1 = fz.FuzzyAditivo(C=self.C).fit(X, y, nomes)
        kmax = n if self.k is None else self.k
        self.subconj = [A for r in range(1, kmax + 1) for A in combinations(range(n), r)]
        Z = self._z(self._parciais(X)); p = len(self.subconj); y = np.asarray(y, float)
        # monotonicidade (Möbius): para cada i e cada B ⊆ N∖{i}: Σ_{C⊆B} β(C ∪ {i}) ≥ 0
        idx = {A: t for t, A in enumerate(self.subconj)}; linhas = []
        for i in range(n):
            resto = [j for j in range(n) if j != i]
            for r in range(len(resto) + 1):
                for B in combinations(resto, r):
                    v = np.zeros(p + 1)
                    for s in range(len(B) + 1):
                        for Cc in combinations(B, s):
                            A = tuple(sorted(Cc + (i,)))
                            if A in idx:
                                v[idx[A]] = 1.0
                    linhas.append(v)
        Amon = np.array(linhas)

        def perda(th):
            b, a = th[:p], th[p]; s = Z @ b + a; q = expit(s)
            ll = np.sum(np.logaddexp(0, s) - y * s) + 0.5 * (b @ b) / self.C
            gz = q - y
            return ll, np.concatenate([Z.T @ gz + b / self.C, [gz.sum()]])

        x0 = np.concatenate([np.full(p, 1.0 / p), [np.log(y.mean() / (1 - y.mean()))]])
        x0[:p] = 0; x0[:n] = 1.0                                   # começa aditivo (só os singletons), que é monótono
        res = minimize(perda, x0, jac=True, method="SLSQP", constraints=[{"type": "ineq", "fun": lambda th: Amon @ th,
                                                                          "jac": lambda th: Amon}],
                       options={"maxiter": 500, "ftol": 1e-9})
        self.beta, self.a = res.x[:p], res.x[p]
        return self

    def predict_proba(self, X):
        q = expit(self._z(self._parciais(X)) @ self.beta + self.a)
        return np.column_stack([1 - q, q])

    def indices(self):
        """Shapley de cada característica e interação de cada par, com a medida normalizada (Σ m = 1)."""
        tot = self.beta.sum()
        if tot <= 1e-12:
            return {}, {}
        m = {A: b / tot for A, b in zip(self.subconj, self.beta)}
        n = len(self.nomes)
        shap = {self.nomes[i]: sum(v / len(A) for A, v in m.items() if i in A) for i in range(n)}
        inter = {(self.nomes[i], self.nomes[j]): sum(v / (len(A) - 1) for A, v in m.items() if i in A and j in A)
                 for i, j in combinations(range(n), 2)}
        return shap, inter


MODELOS = {"triangular + soma": lambda: fz.FuzzyAditivo(), "gaussiana + soma": lambda: FuzzyAditivoGauss(),
           "triangular + Choquet 2-aditiva": lambda: FuzzyChoquet(2), "triangular + Choquet": lambda: FuzzyChoquet(None)}


# ----------------------------------------------------------------------------- uma rodada
def uma_rodada(exp, sem, d):
    tab = pd.read_csv(d / "features.csv.gz"); y = tab.erro.values.astype(int)
    if y.sum() < 10 or (1 - y).sum() < 10:
        return None, [], []
    cand = [c for c in NOSSAS if c in tab]
    oof = {v: np.full(len(y), np.nan) for v in VERSOES}; SH, IN = [], []
    for dobra, (tr, te) in enumerate(StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=sem).split(tab, y)):
        dd = tab.iloc[tr]; feats = fz.selecionar(dd, cand, C.FEATURES_ADITIVO)
        for v in VERSOES:
            mod = MODELOS[v]().fit(dd[feats].values, y[tr], feats)
            oof[v][te] = mod.predict_proba(tab.iloc[te][feats].values)[:, 1]
            if isinstance(mod, FuzzyChoquet):
                sh, it = mod.indices()
                SH += [{"versão": v, "experimento": exp, "semente": sem, "dobra": dobra, "característica": f, "Shapley": s}
                       for f, s in sh.items()]
                IN += [{"versão": v, "experimento": exp, "semente": sem, "dobra": dobra,
                        "par": " × ".join(sorted(par)), "interação": val} for par, val in it.items()]
    ds, arq = C.GRADE_INFO.get(exp, (exp, "?"))
    lin = {"experimento": exp, "dataset": ds, "rede": arq, "semente": sem, "erro da rede": y.mean(), "erros": int(y.sum())}
    for v in VERSOES:
        lin[f"AUROC {v}"] = roc_auc_score(y, oof[v])
        if v != BASE:
            a1, a2, p = delong(y, oof[v], oof[BASE])
            lin[f"dif {v}"] = a1 - a2
            lin[f"resultado {v}"] = "vitória" if (p < 0.05 and a1 > a2) else ("derrota" if (p < 0.05 and a1 < a2) else "empate")
    return lin, SH, IN


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", default=str(C.SAIDA))
    ap.add_argument("--exp", nargs="*", help="só estes experimentos (padrão: toda a grade)")
    ap.add_argument("--jobs", type=int, default=1, help="rodadas em paralelo (CPU)")
    a = ap.parse_args()
    pasta = Path(a.pasta)
    lista = [r for r in rodadas(pasta) if not a.exp or r[0] in a.exp]
    if not lista:
        raise SystemExit(f"Nenhuma rodada em {pasta}/.")
    print(f"{len(lista)} rodadas")

    def roda(r):
        try:
            return uma_rodada(*r)
        except Exception as err:
            print(f"  *** {r[0]} semente {r[1]}: {type(err).__name__}: {err} — pulando"); return None, [], []

    if a.jobs > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(roda)(r) for r in lista)
    else:
        res = []
        for r in lista:
            res.append(roda(r))
            if res[-1][0]:
                l = res[-1][0]
                print(f"  {l['experimento']:28s} {l['semente']:>5}: " + " | ".join(f"{v.split(' + ')[0][:3]}+{v.split(' + ')[1][:9]} {l['AUROC ' + v]:.3f}" for v in VERSOES), flush=True)
    L = pd.DataFrame([r[0] for r in res if r[0]]); SH = pd.DataFrame([x for r in res for x in r[1]]); IN = pd.DataFrame([x for r in res for x in r[2]])

    # resumo
    cel = L.groupby(["dataset", "rede"])[[f"AUROC {v}" for v in VERSOES]].mean()
    for v in VERSOES[1:]:
        cel[f"dif {v}"] = cel[f"AUROC {v}"] - cel[f"AUROC {BASE}"]
    placar = pd.DataFrame({v: L[f"resultado {v}"].value_counts().reindex(["vitória", "empate", "derrota"], fill_value=0)
                           for v in VERSOES[1:]}).T
    placar["diferença média de AUROC"] = [L[f"dif {v}"].mean() for v in VERSOES[1:]]
    placar["maior ganho"] = [L[f"dif {v}"].max() for v in VERSOES[1:]]
    placar["maior perda"] = [L[f"dif {v}"].min() for v in VERSOES[1:]]
    shap = SH.groupby(["versão", "característica"]).Shapley.agg(["mean", "count"]).rename(columns={"mean": "Shapley médio", "count": "dobras"}) \
        if len(SH) else pd.DataFrame()
    inter = IN.groupby(["versão", "par"]).interação.agg(["mean", "count", lambda s: (s > 0.02).mean(), lambda s: (s < -0.02).mean()]) \
        .set_axis(["interação média", "dobras", "fração complementar (>0,02)", "fração redundante (<−0,02)"], axis=1) \
        .sort_values("interação média") if len(IN) else pd.DataFrame()

    arq = pasta / "agregacao.xlsx"
    with pd.ExcelWriter(arq) as xw:
        placar.round(4).to_excel(xw, sheet_name="placar")
        cel.round(4).to_excel(xw, sheet_name="por_combinacao")
        if len(inter):
            inter.round(4).to_excel(xw, sheet_name="interacoes")
        if len(shap):
            shap.round(4).to_excel(xw, sheet_name="shapley")
        L.round(5).to_excel(xw, sheet_name="por_rodada", index=False)

    pd.set_option("display.width", 220)
    print("\nCONTRA O SISTEMA ATUAL (triangular + soma), DeLong p < 0,05 por rodada:")
    print(placar.round(4).to_string())
    if len(inter):
        q = inter.xs("triangular + Choquet 2-aditiva", level="versão") if "triangular + Choquet 2-aditiva" in inter.index.get_level_values(0) else inter
        q = q[q["dobras"] >= 0.2 * q["dobras"].max()]
        print("\nINTERAÇÕES (Choquet 2-aditiva; > 0 = complementares, < 0 = redundantes), pares que aparecem com frequência:")
        print(q.round(3).to_string())
    print(f"\nPlanilha: {arq}")


if __name__ == "__main__":
    main()
