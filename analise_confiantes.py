"""
analise_confiantes.py — "a rede está muito segura, mas errou: dá para perceber?"

Duas etapas, a partir do que o rodar.py já salvou (não roda nada na GPU):
  1. FILTRO: o p serve só para escolher as imagens em que a rede está muito confiante
     (as --frac mais seguras do teste; no binário, as mais longe do limiar, dos dois lados).
     O filtro não usa o rótulo, então não vaza nada.
  2. FUZZY SEM P: dentro desse grupo, o fuzzy aditivo aprende só com as características internas
     (caminho pelos blocos, imagem transformada, vizinhos), por validação cruzada DENTRO do grupo.
     Para comparar: a mesma coisa COM p, e todos os outros métodos (escores já salvos) restritos ao grupo.

    python analise_confiantes.py                          # 30% mais seguras, todos os experimentos
    python analise_confiantes.py --frac 0.2 0.3 0.5       # várias faixas
    python analise_confiantes.py --pasta resultados_teste_rapido

Saída: <pasta>/confiantes.xlsx
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import fuzzy as fz
from leitura import NOSSAS, NOVAS

INTERNAS = [c for c in NOSSAS + NOVAS if c != "p"]  # as internas sem o p (+ as novas, quando a rodada tiver)


def selecionar_sem_p(d, cand, k, max_corr=0.85):
    """As k que mais separam erro de acerto DENTRO do grupo (|AUROC − 0,5|), sem redundância (|ρ| < max_corr)."""
    y = d.erro.values
    forca = {f: abs(roc_auc_score(y, d[f]) - 0.5) if d[f].nunique() > 1 else 0 for f in cand}
    esc = []
    for f in sorted(cand, key=lambda f: -forca[f]):
        if len(esc) == k:
            break
        if forca[f] > 0 and all(abs(spearmanr(d[f], d[g]).correlation) < max_corr for g in esc):
            esc.append(f)
    return esc


def fuzzy_no_grupo(G, cand, k, com_p, semente):
    """Notas fora-da-dobra do fuzzy aditivo dentro do grupo; devolve (notas, features escolhidas em cada dobra)."""
    y = G.erro.values; s = np.full(len(y), np.nan); escolhas = []
    n_dobras = int(min(5, np.bincount(y, minlength=2).min()))
    if n_dobras < 2:
        return s, escolhas
    for tr, te in StratifiedKFold(n_dobras, shuffle=True, random_state=semente).split(G, y):
        d = G.iloc[tr]
        feats = fz.selecionar(d, cand + ["p"], k + 1) if com_p else selecionar_sem_p(d, cand, k)
        if not feats:
            s[te] = y[tr].mean(); continue
        escolhas.append(feats)
        s[te] = fz.FuzzyAditivo().fit(d[feats].values, y[tr], feats).predict_proba(G.iloc[te][feats].values)[:, 1]
    return s, escolhas


def metricas(y, s, frac_revisar=(0.1, 0.2)):
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return {}
    o = np.argsort(-s, kind="stable"); out = {"AUROC": roc_auc_score(y, s), "AUPR do erro": average_precision_score(y, s)}
    for f in frac_revisar:
        n = max(1, int(round(f * len(y))))
        out[f"erros pegos revisando {int(100 * f)}%"] = y[o[:n]].sum() / y.sum()
    return out


def uma_rodada(pasta, frac, k, semente):
    F = pd.read_csv(pasta / "features.csv.gz"); E = pd.read_csv(pasta / "escores.csv.gz")
    corte = np.quantile(F.prox_p, frac)                 # prox_p: menor = mais confiante (binário: longe do limiar)
    m = (F.prox_p <= corte).values
    G, EG = F[m].reset_index(drop=True), E[m].reset_index(drop=True)
    y = G.erro.values
    base = {"imagens no grupo": int(m.sum()), "erros no grupo": int(y.sum()), "taxa de erro no grupo": y.mean(),
            "erros do grupo / erros da rede": y.sum() / max(F.erro.sum(), 1)}
    lin, esc = [], {}
    cand = [c for c in INTERNAS if c in G]
    for nome, com_p in [("NOSSO: fuzzy confiante (sem p)", False), ("NOSSO: fuzzy confiante (com p)", True)]:
        s, e = fuzzy_no_grupo(G, cand, k, com_p, semente); esc[nome] = e
        lin.append({"método": nome, **base, **metricas(y, s)})
    for c in [c for c in EG.columns if c != "erro"]:     # todos os outros, restritos ao grupo
        lin.append({"método": c, **base, **metricas(y, EG[c].values.astype(float))})
    # regras do fuzzy sem p ajustado no grupo inteiro (para ler)
    feats = selecionar_sem_p(G, cand, k) if len(np.unique(y)) == 2 else []
    regras = pd.DataFrame()
    if feats:
        mod = fz.FuzzyAditivo().fit(G[feats].values, y, feats); regras = mod.regras()
    return pd.DataFrame(lin), esc, regras


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", default="resultados")
    ap.add_argument("--frac", nargs="+", type=float, default=[0.3], help="fração mais confiante do teste")
    ap.add_argument("--k", type=int, default=3, help="quantas características internas o fuzzy usa")
    a = ap.parse_args()
    pasta = Path(a.pasta); L, S, R = [], [], []
    rodadas = sorted(p.parent for p in pasta.glob("*/semente_*/escores.csv.gz") if (p.parent / "features.csv.gz").exists())
    if not rodadas:
        raise SystemExit(f"Nenhuma rodada completa em {pasta}/ (precisa de features.csv.gz e escores.csv.gz).")
    for frac in a.frac:
        for p in rodadas:
            exp, sem = p.parent.name, int(p.name.split("_")[1])
            lin, esc, regras = uma_rodada(p, frac, a.k, sem)
            lin.insert(0, "% mais confiantes", frac); lin.insert(0, "semente", sem); lin.insert(0, "experimento", exp)
            L.append(lin)
            for nome, e in esc.items():
                for f in sorted({x for ee in e for x in ee}):
                    S.append({"% mais confiantes": frac, "experimento": exp, "semente": sem, "sistema": nome,
                              "característica": f, "dobras em que entrou": sum(f in ee for ee in e), "dobras": len(e)})
            if len(regras):
                regras.insert(0, "semente", sem); regras.insert(0, "experimento", exp); regras.insert(0, "% mais confiantes", frac)
                R.append(regras)
        print(f"  {int(100 * frac)}% mais confiantes: {len(rodadas)} rodadas", flush=True)
    L = pd.concat(L, ignore_index=True)
    cols = [c for c in L.columns if c.startswith(("AUROC", "AUPR", "erros pegos"))]
    resumo = L.groupby(["% mais confiantes", "experimento", "método"], sort=False)[
        cols + ["imagens no grupo", "erros no grupo", "taxa de erro no grupo", "erros do grupo / erros da rede"]].mean()
    tab = L.pivot_table(index=["% mais confiantes", "método"], columns="experimento", values="AUROC").round(3)
    arq = pasta / "confiantes.xlsx"
    with pd.ExcelWriter(arq) as xw:
        tab.sort_values(tab.columns[0], ascending=False).to_excel(xw, sheet_name="AUROC_no_grupo")
        resumo.round(4).to_excel(xw, sheet_name="resumo")
        L.round(4).to_excel(xw, sheet_name="por_rodada", index=False)
        if S:
            S = pd.DataFrame(S)
            S.groupby(["% mais confiantes", "sistema", "característica"]).apply(
                lambda g: g["dobras em que entrou"].sum() / g["dobras"].sum(), include_groups=False) \
             .rename("% das dobras").round(3).reset_index().sort_values(["% mais confiantes", "sistema", "% das dobras"],
                                                                          ascending=[True, True, False]) \
             .to_excel(xw, sheet_name="caracteristicas", index=False)
        if R:
            pd.concat(R).to_excel(xw, sheet_name="regras_sem_p", index=False)
    nossos = ["NOSSO: fuzzy confiante (sem p)", "NOSSO: fuzzy confiante (com p)", "NOSSO: Fuzzy aditivo",
              "Chen et al. 2019", "Deep kNN", "MC Dropout (desvio)", "Softmax (MSP)"]
    pd.set_option("display.width", 250)
    print("\nAUROC dentro do grupo mais confiante (média das sementes):")
    print(tab[tab.index.get_level_values(1).isin(nossos)].to_string())
    print(f"\nPlanilha: {arq}")


if __name__ == "__main__":
    main()
