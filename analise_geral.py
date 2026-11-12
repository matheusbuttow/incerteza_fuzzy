"""
analise_geral.py — junta TODAS as rodadas da grade (vários datasets × as mesmas 4 redes) e compara o nosso fuzzy com a literatura.
Não usa a GPU: lê o que o rodar.py salvou (escores.csv.gz, features.csv.gz).

O que calcula:
  1. AUROC (detectar o erro da rede) de cada método em cada rodada e em cada célula (dataset × rede, média das sementes);
  2. teste de DeLong em cada rodada: nosso × cada método da literatura (vitória / empate / derrota, p < 0,05);
  3. posto médio de cada método em todas as células + teste de Friedman + diferença crítica de Nemenyi (Demšar, 2006);
  4. melhor e pior caso: as células com maior e menor ganho sobre o softmax e sobre o melhor método da literatura;
  5. o ganho cresce com o ruído dos rótulos? com a taxa de erro da rede? (correlação de Spearman);
  6. por arquitetura e por dataset;
  7. erro confiante: dentro das 25% imagens mais confiantes, o fuzzy treinado só nelas × softmax × literatura;
  8. insights.txt: um resumo em texto com os números principais.

    python analise_geral.py                       # → resultados/geral.xlsx, resultados/insights.txt, resultados/figuras/geral_*.png
    python analise_geral.py --pasta resultados_teste_rapido
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C

NOSSO = "NOSSO: Fuzzy aditivo"
NOSSO_1P = "NOSSO: Fuzzy aditivo (sem vistas)"
# métodos da literatura, por família (o nome é o da coluna em escores.csv.gz)
LIT = {"Softmax (MSP)": "saída", "Entropia do softmax": "saída", "TTA (vistas)": "saída",
       "MC Dropout (desvio)": "saída", "MC Dropout (entropia)": "saída",
       "Mahalanobis": "espaço de características", "Deep kNN": "espaço de características",
       "ConfidNet": "aprende o erro", "Chen et al. 2019": "lê camadas internas", "Font et al. 2025": "lê camadas internas",
       "Gorbett & Blanchard 2022": "lê camadas internas",
       "Aghaeipoor et al. 2023": "fuzzy (outros autores)", "Denoeux 2023 (ENNreg)": "fuzzy (outros autores)",
       "Msaddi e Kumbasar 2026": "fuzzy (outros autores)", "Barna et al. 2026": "fuzzy (outros autores)"}
# q_0,05 de Nemenyi (distribuição da amplitude studentizada / √2), k = 2..20 métodos (Demšar, JMLR 2006)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
       13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}
NOME_REDE = {"resnet32": "ResNet-32", "vgg16": "VGG-16", "mobilenet_v2": "MobileNetV2", "cnn": "CNN pequena"}


# ----------------------------------------------------------------------------- DeLong (Sun e Xu, 2014): AUROCs correlacionadas
def delong(y, s1, s2):
    """p-valor bilateral de AUROC(s1) = AUROC(s2) nas mesmas imagens. y = 1 no erro da rede; nota maior = mais risco."""
    pos = y == 1; m, n = pos.sum(), (~pos).sum()
    X = np.vstack([s1, s2]).astype(float)
    tx = np.vstack([rankdata(x[pos]) for x in X]); ty = np.vstack([rankdata(x[~pos]) for x in X])
    tz = np.vstack([rankdata(x) for x in X])
    auc = (tz[:, pos].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, pos] - tx) / n; v10 = 1 - (tz[:, ~pos] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return auc[0], auc[1], 1.0
    return auc[0], auc[1], 2 * norm.sf(abs(auc[0] - auc[1]) / np.sqrt(var))


# ----------------------------------------------------------------------------- leitura
def rodadas(pasta):
    out = []
    for p in sorted(pasta.glob("*/semente_*/escores.csv.gz")):
        exp = p.parent.parent.name
        if exp in C.GRADE_INFO and (p.parent / "features.csv.gz").exists():
            out.append((exp, int(p.parent.name.split("_")[1]), p.parent))
    return out


def por_rodada(lista, frac_conf):
    import analise_confiantes as ac
    A, W, CF = [], [], []
    for exp, sem, d in lista:
        ds, arq = C.GRADE_INFO[exp]
        E = pd.read_csv(d / "escores.csv.gz"); F = pd.read_csv(d / "features.csv.gz")
        y = E.erro.values.astype(int)
        if y.sum() < 10 or (1 - y).sum() < 10:
            print(f"  (pulando {exp} semente {sem}: só {y.sum()} erros)"); continue
        metodos = [m for m in [NOSSO, NOSSO_1P] + list(LIT) if m in E and E[m].notna().all()]
        base = {"experimento": exp, "dataset": ds, "rede": arq, "semente": sem, "erro da rede": y.mean(),
                "rótulos de treino errados": C.GRADE_DADOS[ds][4], "imagens": len(y), "erros": int(y.sum())}
        for m in metodos:
            A.append({**base, "método": m, "AUROC": roc_auc_score(y, E[m]), "AUPR-erro": average_precision_score(y, E[m])})
        for m in [m for m in metodos if m in LIT]:                              # DeLong: nosso × cada método da literatura
            a1, a2, p = delong(y, E[NOSSO].values, E[m].values)
            W.append({**base, "método": m, "família": LIT[m], "AUROC nosso": a1, "AUROC método": a2, "diferença": a1 - a2, "p": p,
                      "resultado": "vitória" if (p < 0.05 and a1 > a2) else ("derrota" if (p < 0.05 and a1 < a2) else "empate")})
        # erro confiante: as frac_conf mais confiantes (prox_p menor = mais longe da dúvida)
        g = (F.prox_p <= np.quantile(F.prox_p, frac_conf)).values; yg = y[g]
        if yg.sum() >= 20:
            G = F[g].reset_index(drop=True)
            s, _ = ac.fuzzy_no_grupo(G, [c for c in ac.INTERNAS if c in G], 3, False, sem); ok = ~np.isnan(s)
            lin = {**base, "erros no grupo": int(yg.sum()), "taxa de erro no grupo": yg.mean(),
                   "NOSSO: fuzzy só dos confiantes": roc_auc_score(yg[ok], s[ok]) if len(np.unique(yg[ok])) == 2 else np.nan}
            for m in metodos:
                lin[m] = roc_auc_score(yg, E[m].values[g])
            CF.append(lin)
        print(f"  {exp:28s} semente {sem}: erro {y.mean():.3f} | nosso {A[-len(metodos)]['AUROC']:.3f}", flush=True)
    return pd.DataFrame(A), pd.DataFrame(W), pd.DataFrame(CF)


# ----------------------------------------------------------------------------- figuras
def figura_cd(postos, cd, n_celulas, arq):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = postos.sort_values(); k = len(r)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * k + 1.2))
    for i, (m, v) in enumerate(r.items()):
        cor = "#1f5fa8" if m.startswith("NOSSO") else ("#d9622b" if LIT.get(m) == "fuzzy (outros autores)" else "#555555")
        ax.plot([v], [i], "o", color=cor); ax.text(v + 0.12, i, m.replace("NOSSO: ", "nosso: "), va="center", fontsize=8, color=cor)
    ax.plot([r.iloc[0], r.iloc[0] + cd], [-0.9, -0.9], color="k", lw=2); ax.text(r.iloc[0] + cd / 2, -1.3, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlabel(f"posto médio da AUROC em {n_celulas} combinações dataset × rede (1 = melhor)")
    ax.set_xlim(0.5, k + 3); ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(); fig.savefig(arq, dpi=180); plt.close(fig)


def figura_ganho(cel, arq):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    marc = {"resnet32": "o", "vgg16": "s", "mobilenet_v2": "^", "cnn": "D"}
    for a, col, tit in [(ax[0], "ganho sobre o softmax", "nosso − softmax"), (ax[1], "ganho sobre o melhor da literatura", "nosso − melhor da literatura")]:
        for rd, m in marc.items():
            c = cel[cel.rede == rd]
            sc = a.scatter(c["erro da rede"], c[col], c=c["rótulos de treino errados"], cmap="viridis", vmin=0, vmax=0.45, marker=m,
                           s=40, edgecolor="k", lw=0.4, label=NOME_REDE[rd])
        a.axhline(0, color="#888", lw=0.8); a.set_xlabel("taxa de erro da rede"); a.set_ylabel("ganho de AUROC"); a.set_title(tit, fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=7, frameon=False); fig.colorbar(sc, ax=ax, label="rótulos de treino errados")
    fig.savefig(arq, dpi=180, bbox_inches="tight"); plt.close(fig)


def figura_mapa(cel, arq):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = cel.pivot(index="dataset", columns="rede", values="ganho sobre o melhor da literatura")
    M = M.reindex([d for d in C.GRADE_DADOS if d in M.index])[[r for r in C.GRADE_REDES if r in M.columns]]
    v = np.nanmax(np.abs(M.values)) or 0.01
    fig, ax = plt.subplots(figsize=(1.3 * M.shape[1] + 2.5, 0.42 * M.shape[0] + 1.2))
    im = ax.imshow(M.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M.values[i, j]):
                ax.text(j, i, f"{M.values[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([NOME_REDE[r] for r in M.columns], fontsize=8)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([C.GRADE_DADOS[d][3] for d in M.index], fontsize=8)
    ax.set_title("AUROC: nosso − melhor método da literatura (azul = nosso ganha)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout(); fig.savefig(arq, dpi=180); plt.close(fig)


# ----------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", default=str(C.SAIDA))
    ap.add_argument("--frac", type=float, default=0.25, help="fração mais confiante para o erro confiante")
    a = ap.parse_args()
    pasta = Path(a.pasta); lista = rodadas(pasta)
    if not lista:
        raise SystemExit(f"Nenhuma rodada da grade em {pasta}/.")
    print(f"{len(lista)} rodadas da grade encontradas")
    A, W, CF = por_rodada(lista, a.frac)

    # células dataset × rede (média das sementes)
    T = A.pivot_table(index=["dataset", "rede"], columns="método", values="AUROC")
    lit = [m for m in LIT if m in T]
    info = A.groupby(["dataset", "rede"]).agg(**{"erro da rede": ("erro da rede", "mean"), "sementes": ("semente", "nunique"),
                                                   "rótulos de treino errados": ("rótulos de treino errados", "first")})
    cel = info.join(T).reset_index()
    cel["melhor da literatura"] = T[lit].idxmax(axis=1).values
    cel["AUROC melhor da literatura"] = T[lit].max(axis=1).values
    cel["ganho sobre o softmax"] = cel[NOSSO] - cel["Softmax (MSP)"]
    cel["ganho sobre o melhor da literatura"] = cel[NOSSO] - cel["AUROC melhor da literatura"]
    cel["ganho sobre Chen"] = cel[NOSSO] - cel["Chen et al. 2019"] if "Chen et al. 2019" in cel else np.nan
    fz = [m for m in lit if LIT[m] == "fuzzy (outros autores)"]
    cel["ganho sobre o melhor fuzzy de outros autores"] = cel[NOSSO] - T[fz].max(axis=1).values if fz else np.nan

    # postos, Friedman e Nemenyi (só métodos presentes em todas as células)
    met = [m for m in [NOSSO] + lit if T[m].notna().all()]
    R = T[met].rank(axis=1, ascending=False)
    postos = R.mean().sort_values(); k, N = len(met), len(T)
    fried = friedmanchisquare(*[T[m].values for m in met]) if N >= 2 and k >= 3 else None
    cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * N)) if N >= 2 else np.nan

    # vitórias / empates / derrotas contra cada método (DeLong por rodada)
    V = W.groupby(["método", "família"]).resultado.value_counts().unstack(fill_value=0) \
         .reindex(columns=["vitória", "empate", "derrota"], fill_value=0)
    V["rodadas"] = V.sum(axis=1); V["diferença média de AUROC"] = W.groupby(["método", "família"])["diferença"].mean()
    V = V.sort_values("diferença média de AUROC")

    # melhor e pior caso
    cols = ["dataset", "rede", "erro da rede", "rótulos de treino errados", NOSSO, "Softmax (MSP)", "melhor da literatura",
            "AUROC melhor da literatura", "ganho sobre o softmax", "ganho sobre o melhor da literatura"]
    ordem = cel.sort_values("ganho sobre o melhor da literatura", ascending=False)[cols]

    # correlações: o ganho acompanha o ruído e o erro da rede?
    corr = []
    for alvo in ["ganho sobre o softmax", "ganho sobre o melhor da literatura"]:
        for x in ["rótulos de treino errados", "erro da rede"]:
            r = spearmanr(cel[x], cel[alvo]); corr.append({"ganho": alvo, "variável": x, "Spearman ρ": r.correlation, "p": r.pvalue, "células": len(cel)})
    corr = pd.DataFrame(corr)
    por_rede = cel.groupby("rede")[["ganho sobre o softmax", "ganho sobre o melhor da literatura", NOSSO]].mean()
    por_rede["posto médio do nosso"] = R[NOSSO].groupby(level="rede").mean()
    por_ds = cel.groupby("dataset")[["erro da rede", "rótulos de treino errados", "ganho sobre o softmax", "ganho sobre o melhor da literatura"]].mean()
    por_ds["posto médio do nosso"] = R[NOSSO].groupby(level="dataset").mean()

    # erro confiante
    conf = pd.DataFrame()
    if len(CF):
        cm = [c for c in ["NOSSO: fuzzy só dos confiantes", NOSSO, "Softmax (MSP)", "Chen et al. 2019", "Gorbett & Blanchard 2022",
                          "Denoeux 2023 (ENNreg)", "Deep kNN", "MC Dropout (desvio)"] if c in CF]
        conf = CF.groupby(["dataset", "rede"])[["erros no grupo", "taxa de erro no grupo"] + cm].mean()
        litc = [m for m in lit if m in CF]
        conf["melhor da literatura"] = CF.groupby(["dataset", "rede"])[litc].mean().max(axis=1)

    # ----- planilha
    arq = pasta / "geral.xlsx"
    with pd.ExcelWriter(arq) as xw:
        cel.round(4).to_excel(xw, sheet_name="celulas", index=False)
        ordem.round(4).to_excel(xw, sheet_name="melhor_e_pior_caso", index=False)
        V.round(4).to_excel(xw, sheet_name="contra_literatura")
        pd.DataFrame({"posto médio": postos, "família": [LIT.get(m, "nosso") for m in postos.index]}).round(3) \
          .to_excel(xw, sheet_name="postos")
        corr.round(4).to_excel(xw, sheet_name="ganho_vs_ruido", index=False)
        por_rede.round(4).to_excel(xw, sheet_name="por_rede"); por_ds.round(4).to_excel(xw, sheet_name="por_dataset")
        if len(conf):
            conf.round(4).to_excel(xw, sheet_name="erro_confiante")
        A.round(4).to_excel(xw, sheet_name="por_rodada", index=False); W.round(5).to_excel(xw, sheet_name="delong", index=False)

    # ----- figuras
    (pasta / "figuras").mkdir(exist_ok=True)
    try:
        if N >= 2:
            figura_cd(postos, cd, N, pasta / "figuras" / "geral_postos.png")
        figura_ganho(cel, pasta / "figuras" / "geral_ganho_vs_erro.png")
        figura_mapa(cel, pasta / "figuras" / "geral_mapa.png")
    except Exception as err:
        print("(figuras não geradas:", err, ")")

    # ----- insights em texto
    L = []; f3 = lambda v: f"{v:.3f}"
    L.append(f"RODADAS: {len(A.groupby(['experimento', 'semente']))} redes treinadas | {N} combinações dataset × rede | "
             f"{cel.dataset.nunique()} datasets | {cel.rede.nunique()} redes | sementes por combinação: {cel.sementes.min()}–{cel.sementes.max()}")
    melhor = (T[met].idxmax(axis=1) == NOSSO).sum()
    L.append(f"\n1. O NOSSO É O MELHOR em {melhor} de {N} combinações (entre {k} métodos: nosso + {k - 1} da literatura).")
    L.append(f"   Posto médio: nosso {f3(postos[NOSSO])} (1 = melhor). Os 3 seguintes: " +
             ", ".join(f"{m} {f3(v)}" for m, v in postos.drop(NOSSO).head(3).items()) + ".")
    if fried is not None:
        L.append(f"   Friedman: χ² = {fried.statistic:.1f}, p = {fried.pvalue:.2g}. Diferença crítica de Nemenyi (α = 0,05): {cd:.2f}.")
        dif = [m for m in met if m != NOSSO and postos[m] - postos[NOSSO] > cd]
        L.append(f"   O nosso é significativamente melhor (Nemenyi) que {len(dif)} de {k - 1} métodos; não se distingue de: " +
                 (", ".join(m for m in met if m != NOSSO and m not in dif) or "nenhum") + ".")
    vt = V[["vitória", "empate", "derrota"]].sum()
    L.append(f"\n2. CONTRA CADA MÉTODO (DeLong, p < 0,05, por rodada): {vt['vitória']} vitórias, {vt['empate']} empates, {vt['derrota']} derrotas.")
    for (m, fam), r in V.iterrows():
        L.append(f"   {m:28s} [{fam}]: {r['vitória']}V {r['empate']}E {r['derrota']}D | diferença média {r['diferença média de AUROC']:+.3f}")
    b, w = ordem.iloc[0], ordem.iloc[-1]
    L.append(f"\n3. MELHOR CASO: {C.GRADE_DADOS[b.dataset][3]} + {NOME_REDE[b.rede]}: nosso {f3(b[NOSSO])} × melhor da literatura "
             f"{f3(b['AUROC melhor da literatura'])} ({b['melhor da literatura']}), ganho {b['ganho sobre o melhor da literatura']:+.3f}; "
             f"sobre o softmax {b['ganho sobre o softmax']:+.3f}.")
    L.append(f"   PIOR CASO: {C.GRADE_DADOS[w.dataset][3]} + {NOME_REDE[w.rede]}: nosso {f3(w[NOSSO])} × melhor da literatura "
             f"{f3(w['AUROC melhor da literatura'])} ({w['melhor da literatura']}), ganho {w['ganho sobre o melhor da literatura']:+.3f}; "
             f"sobre o softmax {w['ganho sobre o softmax']:+.3f}.")
    perde = ordem[ordem["ganho sobre o melhor da literatura"] < 0]
    if len(perde):
        L.append(f"   Combinações em que algum método da literatura passa o nosso ({len(perde)}): " +
                 "; ".join(f"{C.GRADE_DADOS[r.dataset][3]}+{NOME_REDE[r.rede]} ({r['melhor da literatura']}, {r['ganho sobre o melhor da literatura']:+.3f})"
                           for r in perde.itertuples(index=False) if True))
    L.append("\n4. O GANHO ACOMPANHA O RUÍDO? (Spearman nas combinações)")
    for r in corr.itertuples(index=False):
        L.append(f"   {r[0]} × {r[1]}: ρ = {r[2]:+.2f} (p = {r[3]:.2g})")
    lim = cel.groupby(cel["rótulos de treino errados"] > 0)[["ganho sobre o softmax", "ganho sobre o melhor da literatura"]].mean()
    for ruid, r in lim.iterrows():
        L.append(f"   {'com' if ruid else 'sem'} ruído nos rótulos: ganho médio sobre o softmax {r.iloc[0]:+.3f}, sobre o melhor da literatura {r.iloc[1]:+.3f}")
    L.append("\n5. POR REDE (média nas combinações):")
    for rd, r in por_rede.iterrows():
        L.append(f"   {NOME_REDE.get(rd, rd):12s} nosso {f3(r[NOSSO])} | ganho sobre softmax {r['ganho sobre o softmax']:+.3f} | "
                 f"sobre o melhor da literatura {r['ganho sobre o melhor da literatura']:+.3f} | posto {r['posto médio do nosso']:.2f}")
    L.append("\n6. POR DATASET:")
    for d, r in por_ds.iterrows():
        L.append(f"   {C.GRADE_DADOS[d][3]:22s} erro da rede {r['erro da rede']:.3f} | ganho sobre softmax {r['ganho sobre o softmax']:+.3f} | "
                 f"sobre o melhor da literatura {r['ganho sobre o melhor da literatura']:+.3f}")
    if fz:
        g = cel["ganho sobre o melhor fuzzy de outros autores"]
        L.append(f"\n7. CONTRA OS FUZZY DE OUTROS AUTORES ({', '.join(fz)}): ganho médio {g.mean():+.3f}, "
                 f"o nosso é melhor em {(g > 0).sum()} de {len(g)} combinações.")
        if "Denoeux 2023 (ENNreg)" in fz:
            L.append("   Atenção: o ENNreg recebe as NOSSAS características como entrada; perto do nosso = a força está no que o sistema lê.")
    if len(conf):
        ok = conf.dropna(subset=["NOSSO: fuzzy só dos confiantes"])
        L.append(f"\n8. ERRO CONFIANTE ({int(100 * a.frac)}% mais confiantes, combinações com ≥ 20 erros no grupo: {len(ok)}):")
        if len(ok):
            L.append(f"   softmax {f3(ok['Softmax (MSP)'].mean())} | melhor da literatura {f3(ok['melhor da literatura'].mean())} | "
                     f"nosso (fuzzy só dos confiantes) {f3(ok['NOSSO: fuzzy só dos confiantes'].mean())}; "
                     f"o nosso passa o melhor da literatura em {(ok['NOSSO: fuzzy só dos confiantes'] > ok['melhor da literatura']).sum()} de {len(ok)}.")
            semg = cel[cel["rótulos de treino errados"] == 0]
            L.append(f"   (Em datasets limpos o grupo costuma ter poucos erros: {len(conf) - len(ok)} combinações ficaram de fora.)" if len(semg) else "")
    if NOSSO_1P in T:
        g1 = (T[NOSSO_1P] - T["Softmax (MSP)"]).mean()
        L.append(f"\n9. VERSÃO DE UMA PASSADA (sem as vistas): ganho médio sobre o softmax {g1:+.3f} "
                 f"(com vistas: {cel['ganho sobre o softmax'].mean():+.3f}).")
    txt = "\n".join(l for l in L if l is not None)
    (pasta / "insights.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\nPlanilha: {arq}\nResumo: {pasta / 'insights.txt'}\nFiguras: {pasta / 'figuras'}/geral_*.png")


if __name__ == "__main__":
    main()
