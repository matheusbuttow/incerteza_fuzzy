"""
fuzzy_t2.py — Wang-Mendel tipo 1 e tipo 2 intervalar, e o teste "treina em umas redes, aplica em outra".

TIPO 1: cada feature tem 3 termos (BAIXA, MÉDIA, ALTA), triângulos com picos nos quantis 5%, 50%, 95%.

TIPO 2 (envelope entre redes): cada rede de origem tem os seus próprios picos. Para cada termo,
  pertinência inferior = a menor entre as redes, superior = a maior.
  A faixa entre as duas (a "pegada de incerteza") é exatamente a discordância entre as redes sobre
  o que é, por exemplo, "taxa_trocas ALTA".

INFERÊNCIA DO TIPO 2: cada regra dispara com uma força num intervalo [f_inf, f_sup]
  (produto das pertinências inferiores / superiores). A nota de erro é
      s = E / (E + A),   E = Σ força·CF das regras ERRO,   A = Σ força·CF das regras ACERTO.
  s cresce com E e decresce com A, então os extremos do intervalo saem direto (sem iteração de Karnik-Mendel):
      s_inf = E_inf / (E_inf + A_sup),   s_sup = E_sup / (E_sup + A_inf).
  Nota pontual = a nota com as pertinências do MEIO da faixa (fica sempre dentro do intervalo).
  Largura = o quanto as redes discordam naquele caso; [0, 1] = discordância total (termos sem sobreposição).

TRÊS ZONAS: limiar τ escolhido nas redes de origem. alarme: s_inf > τ · seguro: s_sup < τ · incerto: o resto.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
TERMOS = ["BAIXA", "MÉDIA", "ALTA"]


# ============================================================================ pertinências
def centros(X):
    """Picos dos 3 termos por feature (J, 3): quantis 5%, 50%, 95%, estritamente crescentes."""
    C = np.nanquantile(np.asarray(X, float), [0.05, 0.5, 0.95], axis=0).T.copy()
    for j in range(len(C)):
        for k in (1, 2):
            C[j, k] = max(C[j, k], C[j, k - 1] + 1e-6 * (abs(C[j, k - 1]) + 1))
    return C


def pert_t1(X, C):
    """Pertinências (n, J, 3) com triângulos que somam 1."""
    X = np.asarray(X, float); n, J = X.shape
    M = np.zeros((n, J, 3))
    for j in range(J):
        c = C[j]; x = np.nan_to_num(X[:, j], nan=c[1])
        M[:, j, 0] = x <= c[0]; M[:, j, 2] = x >= c[2]
        for k in (0, 1):
            d = (x >= c[k]) & (x < c[k + 1]); t = (x[d] - c[k]) / (c[k + 1] - c[k])
            M[d, j, k] = 1 - t; M[d, j, k + 1] = t
    return M


def pert_t2(X, lista_C):
    """Envelope das pertinências tipo 1 de várias redes: (inferior, superior), cada um (n, J, 3)."""
    Ms = np.stack([pert_t1(X, C) for C in lista_C])
    return Ms.min(0), Ms.max(0)


# ============================================================================ Wang-Mendel
class WangMendel:
    """Regras aprendidas a partir de pertinências já calculadas (tipo 1, ou o ponto médio do tipo 2).
    Cada exemplo propõe a regra com o termo de maior grau em cada feature; ERRO/ACERTO e o fator de certeza (CF)
    vêm de todos os exemplos compatíveis. Pesos: classes balanceadas DENTRO de cada rede de origem e
    cada rede com o mesmo peso total (uma rede não domina só por ter mais erros)."""

    def __init__(self, min_suporte=0.01, min_cf=0.1):
        self.min_suporte, self.min_cf = min_suporte, min_cf

    def fit(self, M, y, nomes, grupos=None):
        y = np.asarray(y).astype(int); n = len(y)
        grupos = np.zeros(n, int) if grupos is None else np.asarray(grupos)
        self.nomes = list(nomes)
        w = np.zeros(n); G = np.unique(grupos)
        for g in G:
            m = grupos == g
            for c in (0, 1):
                mc = m & (y == c)
                if mc.any():
                    w[mc] = 1.0 / (2 * mc.sum() * len(G))
        ant_por_ex = M.argmax(2)                                    # (n, J)
        self.regras_ = []
        for ant in {tuple(a) for a in ant_por_ex}:
            comp = self._forca_uma(M, ant)
            sup = comp.sum() / n
            b = np.array([(comp * w)[y == 0].sum(), (comp * w)[y == 1].sum()])
            if sup < self.min_suporte or b.sum() <= EPS:
                continue
            c = int(b.argmax()); cf = (b[c] - b[1 - c]) / b.sum()
            if cf >= self.min_cf:
                self.regras_.append({"ant": ant, "cons": c, "cf": float(cf), "suporte": float(sup)})
        self.regras_.sort(key=lambda r: -r["suporte"] * r["cf"])
        return self

    @staticmethod
    def _forca_uma(M, ant):
        return np.prod(M[:, np.arange(len(ant)), list(ant)], axis=1)

    def _somas(self, M):
        E = np.zeros(len(M)); A = np.zeros(len(M))
        for r in self.regras_:
            f = self._forca_uma(M, r["ant"]) * r["cf"]
            if r["cons"] == 1:
                E += f
            else:
                A += f
        return E, A

    def nota_t1(self, M):
        E, A = self._somas(M)
        return np.where(E + A > EPS, E / np.maximum(E + A, EPS), 0.5)

    def nota_t2(self, M_inf, M_sup):
        """(s_inf, s_sup, s_meio) para pertinências intervalares. s_meio usa as pertinências do meio da faixa,
        as mesmas com que as regras foram aprendidas."""
        E_i, A_i = self._somas(M_inf)
        E_s, A_s = self._somas(M_sup)
        s_inf = np.where(E_i + A_s > EPS, E_i / np.maximum(E_i + A_s, EPS), 0.5)
        s_sup = np.where(E_s + A_i > EPS, E_s / np.maximum(E_s + A_i, EPS), 0.5)
        s_sup = np.maximum(s_sup, s_inf)
        s_meio = np.clip(self.nota_t1((M_inf + M_sup) / 2), s_inf, s_sup)
        return s_inf, s_sup, s_meio

    @property
    def n_regras(self):
        return len(self.regras_)

    def tabela(self):
        return pd.DataFrame([{"regra": "SE " + " E ".join(f"{self.nomes[j]} é {TERMOS[k]}" for j, k in enumerate(r["ant"]))
                              + (" ENTÃO ERRO" if r["cons"] else " ENTÃO ACERTO"),
                              "certeza (CF)": round(r["cf"], 3), "suporte": round(r["suporte"], 3)} for r in self.regras_])


def tabela_fou(lista_C, nomes, rotulos):
    """Onde fica o pico de cada termo em cada rede, e a faixa (mín–máx) que vira a incerteza do tipo 2."""
    lin = []
    Cs = np.stack(lista_C)                                          # (S, J, 3)
    for j, f in enumerate(nomes):
        for k, t in enumerate(TERMOS):
            d = {"feature": f, "termo": t}
            for s, r in enumerate(rotulos):
                d[f"pico em {r}"] = Cs[s, j, k]
            d["faixa (mín)"], d["faixa (máx)"] = Cs[:, j, k].min(), Cs[:, j, k].max()
            d["largura relativa"] = (Cs[:, j, k].max() - Cs[:, j, k].min()) / (abs(Cs[:, j, 2] - Cs[:, j, 0]).mean() + EPS)
            lin.append(d)
    return pd.DataFrame(lin)


# ============================================================================ seleção e métricas
def selecionar(tab, cand, k=3, forcar=("prox_p",), max_corr=0.85):
    """Top-k por AUROC (para detectar o erro), sem features muito correlacionadas. `forcar` entram sempre."""
    def forca(f):
        return abs(roc_auc_score(tab.erro, tab[f]) - 0.5) if tab[f].nunique() > 1 and tab.erro.nunique() > 1 else 0
    esc = [f for f in forcar if f in cand]
    for f in sorted(cand, key=lambda f: -forca(f)):
        if len(esc) >= k:
            break
        if f not in esc and all(abs(spearmanr(tab[f], tab[g]).correlation) < max_corr for g in esc):
            esc.append(f)
    return esc


def auroc_alem_de_p(y, s, prox_p, faixas=5):
    """AUROC dentro de faixas de P parecido (média ponderada): o que a nota traz além de P."""
    y, s = np.asarray(y), np.asarray(s)
    bins = pd.qcut(pd.Series(prox_p).rank(method="first"), faixas, labels=False).values
    vals, pesos = [], []
    for b in range(faixas):
        m = bins == b
        if m.any() and y[m].min() != y[m].max():
            vals.append(roc_auc_score(y[m], s[m])); pesos.append(m.sum())
    return float(np.average(vals, weights=pesos)) if vals else np.nan


def aurc(y, s):
    """Área sob a curva risco-cobertura: aceita primeiro as imagens com menor nota de erro. Menor = melhor."""
    o = np.argsort(s, kind="stable"); e = np.asarray(y)[o]
    return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())


def metricas(y, s, prox_p):
    if len(np.unique(y)) < 2:
        return {"AUROC": np.nan, "AUROC além de P": np.nan, "AUPR do erro": np.nan, "AURC (menor=melhor)": np.nan}
    return {"AUROC": roc_auc_score(y, s), "AUROC além de P": auroc_alem_de_p(y, s, prox_p),
            "AUPR do erro": average_precision_score(y, s), "AURC (menor=melhor)": aurc(y, s)}


def ic_diferenca(y, s1, s2, n_boot=1000, seed=0):
    """IC 95% bootstrap (pareado) da diferença de AUROC s1 − s2."""
    r = np.random.default_rng(seed); y = np.asarray(y); d = []
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    for _ in range(n_boot):
        i = r.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return (np.percentile(d, 2.5), np.percentile(d, 97.5)) if d else (np.nan, np.nan)


# ============================================================================ fuzzy aditivo
class FuzzyAditivo:
    """Regras de UMA condição cada, somadas: 'SE taxa_trocas é ALTA → +1,3 no risco de erro (log-odds)'.
    Pesos aprendidos por regressão logística sobre os graus de pertinência (mesmo balanceamento do Wang-Mendel).
    No tipo 2 a nota é linear nas pertinências, então o intervalo sai exato:
      z_inf = b + Σ (w>0 ? w·μ_inf : w·μ_sup),   z_sup = b + Σ (w>0 ? w·μ_sup : w·μ_inf)."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, M, y, nomes, grupos=None):
        y = np.asarray(y).astype(int); n = len(y); self.nomes = list(nomes)
        grupos = np.zeros(n, int) if grupos is None else np.asarray(grupos)
        w = np.ones(n); G = np.unique(grupos)
        for g in G:
            for c in (0, 1):
                mc = (grupos == g) & (y == c)
                if mc.any():
                    w[mc] = n / (2 * mc.sum() * len(G))
        self.ok = len(np.unique(y)) == 2
        if self.ok:
            self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(M.reshape(n, -1), y, sample_weight=w)
            self.w, self.b = self.lr.coef_[0], self.lr.intercept_[0]
        return self

    def nota_t1(self, M):
        if not self.ok:
            return np.full(len(M), 0.5)
        return 1 / (1 + np.exp(-(M.reshape(len(M), -1) @ self.w + self.b)))

    def nota_t2(self, M_inf, M_sup):
        if not self.ok:
            h = np.full(len(M_inf), 0.5); return h, h, h
        a, s = M_inf.reshape(len(M_inf), -1), M_sup.reshape(len(M_sup), -1)
        pos = self.w > 0
        z_i = a[:, pos] @ self.w[pos] + s[:, ~pos] @ self.w[~pos] + self.b
        z_s = s[:, pos] @ self.w[pos] + a[:, ~pos] @ self.w[~pos] + self.b
        sig = lambda z: 1 / (1 + np.exp(-z))
        s_i, s_s = sig(z_i), sig(z_s)
        return s_i, s_s, np.clip(self.nota_t1((M_inf + M_sup) / 2), s_i, s_s)

    def tabela(self):
        if not self.ok:
            return pd.DataFrame()
        lin = [{"regra": f"SE {f} é {TERMOS[k]}", "peso no risco de erro": round(float(self.w[3 * j + k]), 3),
                "efeito": "aumenta" if self.w[3 * j + k] > 0.05 else ("diminui" if self.w[3 * j + k] < -0.05 else "neutro")}
               for j, f in enumerate(self.nomes) for k in range(3)]
        lin.append({"regra": "(base)", "peso no risco de erro": round(float(self.b), 3), "efeito": ""})
        return pd.DataFrame(lin)


# ============================================================================ o experimento
REFERENCIAS = ["Só P", "Regressão logística dentro (referência)", "Regressão logística transferida", "Boosting transferido"]
VARIANTES = ["tipo 1 dentro (referência)", "tipo 1 transferido", "tipo 1 transferido + termos no alvo",
             "tipo 2 transferido", "tipo 2 transferido + alvo no envelope"]
FAMILIAS = ["Wang-Mendel", "Fuzzy aditivo"]
METODOS = REFERENCIAS + [f"{fam}: {v}" for fam in FAMILIAS for v in VARIANTES]
PRINCIPAIS = REFERENCIAS + [f"Fuzzy aditivo: {v}" for v in VARIANTES]      # o que vai para o artigo
INTERNOS = [f"Wang-Mendel: {v}" for v in VARIANTES]                        # só para consulta (fora do artigo)


def _pesos(y, g):
    w = np.ones(len(y)); G = np.unique(g)
    for gg in G:
        for c in (0, 1):
            m = (g == gg) & (y == c)
            if m.any():
                w[m] = len(y) / (2 * m.sum() * len(G))
    return w


def _zonas(alvo, sistema, y, s_inf, s_sup, tau):
    out = []
    for nome_z, mz in [("alarme", s_inf > tau), ("incerto", (s_inf <= tau) & (s_sup >= tau)), ("seguro", s_sup < tau)]:
        if nome_z == "incerto" and np.allclose(s_inf, s_sup, equal_nan=True):
            continue
        out.append({"rede alvo": alvo, "sistema": sistema, "zona": nome_z, "imagens": int(mz.sum()),
                    "% das imagens": mz.mean(), "taxa de erro da rede na zona": y[mz].mean() if mz.any() else np.nan,
                    "erros na zona": int(y[mz].sum()), "% dos erros": y[mz].sum() / max(y.sum(), 1)})
    return out


def deixa_uma_fora(tabs, cand, k_wm=3, k_ad=5, forcar=("prox_p",), q_alarme=0.2, n_boot=1000, log=print):
    """Para cada rede ALVO: os sistemas são aprendidos nas OUTRAS redes e aplicados no alvo.
    O teste de cada rede é dividido em duas metades por lesão: aprende numa metade e avalia na outra (e troca),
    então nenhuma imagem é usada para aprender e avaliar ao mesmo tempo.
    Duas famílias fuzzy, cada uma em tipo 1 e tipo 2: Wang-Mendel (regras com k_wm condições) e fuzzy aditivo
    (regras de uma condição somadas, k_ad features, o mesmo conjunto da regressão logística e do boosting).
    'Termos no alvo' e 'alvo no envelope' usam só as features do alvo, nunca os rótulos dele."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    nomes = list(tabs)
    linhas, zonas, regras, fous, preds = [], [], {}, {}, {}
    for alvo in nomes:
        fontes = [s for s in nomes if s != alvo]
        T = tabs[alvo]; n = len(T)
        nota = {m: np.full(n, np.nan) for m in METODOS}
        iv = {fam: [np.full(n, np.nan) for _ in range(3)] for fam in FAMILIAS}      # s_inf, s_sup, tau
        escolhas = []
        for f in (0, 1):
            te = (T.metade == 1 - f).values
            fonte_tr = pd.concat([tabs[s][tabs[s].metade == f].assign(_g=i) for i, s in enumerate(fontes)], ignore_index=True)
            alvo_tr = T[T.metade == f]
            ys, gs = fonte_tr.erro.values, fonte_tr._g.values
            nota["Só P"][te] = T.loc[te, "prox_p"].values
            f_ad = selecionar(fonte_tr, cand, k_ad, forcar); f_wm = selecionar(fonte_tr, cand, k_wm, forcar)
            escolhas.append((f_wm, f_ad))
            # referências não fuzzy (k_ad features)
            Xs, Xa, Xin = fonte_tr[f_ad].values, T.loc[te, f_ad].values, alvo_tr[f_ad].values
            def lr_fit(X, y, w=None):
                return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(
                    X, y, logisticregression__sample_weight=w) if len(np.unique(y)) == 2 else None
            prob = lambda mdl, X: mdl.predict_proba(X)[:, 1] if mdl is not None else np.full(len(X), 0.5)
            nota["Regressão logística dentro (referência)"][te] = prob(lr_fit(Xin, alvo_tr.erro.values, _pesos(alvo_tr.erro.values, np.zeros(len(alvo_tr)))), Xa)
            nota["Regressão logística transferida"][te] = prob(lr_fit(Xs, ys, _pesos(ys, gs)), Xa)
            if len(np.unique(ys)) == 2:
                gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)
                nota["Boosting transferido"][te] = gb.fit(Xs, ys, sample_weight=_pesos(ys, gs)).predict_proba(Xa)[:, 1]
            else:
                nota["Boosting transferido"][te] = 0.5
            # as duas famílias fuzzy
            for fam, feats, Cls in [("Wang-Mendel", f_wm, WangMendel), ("Fuzzy aditivo", f_ad, FuzzyAditivo)]:
                Xs, Xa, Xin = fonte_tr[feats].values, T.loc[te, feats].values, alvo_tr[feats].values
                Ca = centros(Xin)
                nota[f"{fam}: tipo 1 dentro (referência)"][te] = Cls().fit(pert_t1(Xin, Ca), alvo_tr.erro.values, feats).nota_t1(pert_t1(Xa, Ca))
                C1 = centros(Xs)
                m1 = Cls().fit(pert_t1(Xs, C1), ys, feats, gs)
                nota[f"{fam}: tipo 1 transferido"][te] = m1.nota_t1(pert_t1(Xa, C1))
                nota[f"{fam}: tipo 1 transferido + termos no alvo"][te] = m1.nota_t1(pert_t1(Xa, centros(Xa)))
                lista_C = [centros(fonte_tr[fonte_tr._g == i][feats].values) for i in range(len(fontes))]
                Mi, Ms = pert_t2(Xs, lista_C)
                m2 = Cls().fit((Mi + Ms) / 2, ys, feats, gs)
                s_i, s_s, s_m = m2.nota_t2(*pert_t2(Xa, lista_C))
                nota[f"{fam}: tipo 2 transferido"][te] = s_m
                nota[f"{fam}: tipo 2 transferido + alvo no envelope"][te] = m2.nota_t2(*pert_t2(Xa, lista_C + [centros(Xa)]))[2]
                iv[fam][0][te], iv[fam][1][te] = s_i, s_s
                iv[fam][2][te] = np.quantile(m2.nota_t2(Mi, Ms)[2], 1 - q_alarme)      # limiar escolhido nas FONTES
                if f == 0:
                    regras[(alvo, fam)] = m2.tabela()
                    fous[(alvo, fam)] = tabela_fou(lista_C, feats, fontes)
        y, pp = T.erro.values, T.prox_p.values
        auc = lambda s: roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
        for m in METODOS:
            d = {"rede alvo": alvo, "aprendido em": " + ".join(fontes), "método": m, **metricas(y, nota[m], pp)}
            if ": tipo 2" in m:
                ref = nota[m.split(":")[0] + ": tipo 1 transferido"]
                lo, hi = ic_diferenca(y, nota[m], ref, n_boot)
                d["ganho de AUROC sobre o tipo 1 transferido"] = auc(nota[m]) - auc(ref)
                d["IC 95% do ganho"] = f"[{lo:+.3f}, {hi:+.3f}]"
            linhas.append(d)
        for fam in FAMILIAS:
            zonas += _zonas(alvo, f"{fam} tipo 2", y, *iv[fam])
        p = T[["image_id", "metade", "erro", "p", "prox_p"]].copy()
        for m in METODOS:
            p[m] = nota[m]
        for fam in FAMILIAS:
            p[f"{fam} t2 inf"], p[f"{fam} t2 sup"] = iv[fam][0], iv[fam][1]
            p[f"{fam} t2 largura"] = iv[fam][1] - iv[fam][0]
        preds[alvo] = p
        log(f"alvo {alvo}: WM {escolhas[0][0]} | aditivo/LR {escolhas[0][1]}\n   " + " | ".join(
            f"{m.replace('transferido', 'transf.')}: {auc(nota[m]):.3f}" for m in
            ["Só P", "Regressão logística transferida", "Wang-Mendel: tipo 1 transferido", "Wang-Mendel: tipo 2 transferido",
             "Fuzzy aditivo: tipo 1 transferido", "Fuzzy aditivo: tipo 2 transferido"]))
    return pd.DataFrame(linhas), pd.DataFrame(zonas), regras, fous, preds
