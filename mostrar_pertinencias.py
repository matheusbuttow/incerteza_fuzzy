"""
mostrar_pertinencias.py — desenha as funções de pertinência do fuzzy aditivo final (o das regras do relatório).

Para cada feature escolhida, duas linhas:
  em cima:  os 3 termos (BAIXA, MÉDIA, ALTA) em triângulo; a cor é o peso da regra
            (laranja = a regra aumenta o risco de erro, azul = diminui, cinza = quase neutra);
  embaixo:  quantos pontos de risco essa feature dá para cada valor (soma pertinência × peso),
            com a distribuição dos acertos (azul) e dos erros (vermelho) da rede ao fundo.

Exemplos:
  python mostrar_pertinencias.py                          # HAM completo, semente 42
  python mostrar_pertinencias.py --exp cifar10_ruido30 --semente 7
  python mostrar_pertinencias.py --todos                  # todos os experimentos de uma vez (+ resumo_regras.csv)

Saída: resultados/<exp>/semente_<s>/pertinencias.png
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import fuzzy as fz
from leitura import NOSSAS
from mostrar_casos import NOMES

warnings.filterwarnings("ignore")


def cor(w):
    return "#eb6834" if w > 0.05 else ("#2a78d6" if w < -0.05 else "#999999")


def desenhar(exp, semente, saida=None):
    pasta = Path(saida or C.SAIDA) / exp / f"semente_{semente}"
    tab = pd.read_csv(pasta / "features.csv.gz")
    info = json.loads((pasta / "info_rede.json").read_text())
    fixos = {"p": [0.0, info["limiar"], 1.0]} if info.get("limiar") is not None else None

    feats = fz.selecionar(tab, [c for c in NOSSAS if c in tab], C.FEATURES_ADITIVO)
    ad = fz.FuzzyAditivo(fixos).fit(tab[feats].values, tab.erro.values, feats)
    w, b = ad.pesos(), ad.lr.intercept_[0]

    fig, axs = plt.subplots(2, len(feats), figsize=(4.2 * len(feats), 6.2), sharex="col",
                            gridspec_kw={"height_ratios": [1, 1.1]})
    for j, f in enumerate(feats):
        c = ad.part.centros[j]; v = tab[f].values.astype(float); v = v[np.isfinite(v)]
        lo, hi = (0.0, 1.0) if f == "p" else (min(c[0], np.quantile(v, 0.01)), max(c[2], np.quantile(v, 0.99)))
        pad = 0.05 * (hi - lo) or 1.0
        x = np.linspace(lo - pad, hi + pad, 400)
        X = np.full((len(x), len(feats)), np.nan); X[:, j] = x
        M = ad.part.pertinencias(X)[j]; wj = w[3 * j: 3 * j + 3]

        a1 = axs[0, j]
        for k in range(3):
            a1.plot(x, M[:, k], color=cor(wj[k]), lw=2.5)
            a1.fill_between(x, M[:, k], color=cor(wj[k]), alpha=0.12)
            perto = k > 0 and (c[k] - c[k - 1]) < 0.12 * (hi - lo + 2 * pad)       # rótulos colados: sobe um
            a1.text(c[k], 1.06 + 0.3 * perto, f"{fz.TERMOS[k]}\n{wj[k]:+.2f}", ha="center", va="bottom", fontsize=9, color=cor(wj[k]),
                    fontweight="bold")
        a1.set_ylim(0, 1.62); a1.set_yticks([0, 0.5, 1])
        a1.set_title(NOMES.get(f, f), fontsize=11, pad=4)
        if j == 0:
            a1.set_ylabel("pertinência")

        a2 = axs[1, j]; h = a2.twinx()
        bins = np.linspace(lo - pad, hi + pad, 40)
        for e, cc, nm in [(0, "#2a78d6", "acertos da rede"), (1, "#b8325a", "erros da rede")]:
            h.hist(np.clip(tab.loc[tab.erro == e, f].dropna(), bins[0], bins[-1]), bins=bins, density=True,
                   color=cc, alpha=0.25, label=nm)
        h.set_yticks([])
        a2.set_zorder(h.get_zorder() + 1); a2.patch.set_visible(False)
        a2.plot(x, M @ wj, color="#0B2545", lw=2.5, label="pontos de risco")
        a2.axhline(0, color="#888", lw=0.8)
        if f == "p" and fixos:
            for ax in (a1, a2):
                ax.axvline(fixos["p"][1], color="#555", ls="--", lw=1)
            a2.text(fixos["p"][1], a2.get_ylim()[1], " limiar", fontsize=8, va="top", color="#555")
        a2.set_xlabel(f"valor de '{f}'", fontsize=9)
        if j == 0:
            a2.set_ylabel("pontos de risco\n(+ aumenta, − diminui)")
            l1, n1 = a2.get_legend_handles_labels(); l2, n2 = h.get_legend_handles_labels()
            a2.legend(l1 + l2, n1 + n2, fontsize=8, loc="best", frameon=False)
        for ax in (a1, a2):
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Fuzzy aditivo — funções de pertinência e peso de cada regra  ({exp}, semente {semente})\n"
                 f"risco de erro = sigmoide( base {b:+.2f} + soma dos pontos de risco das {len(feats)} features )",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(pasta / "pertinencias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    r = ad.regras(); r.insert(0, "experimento", exp)
    print(f"\n== {exp} | semente {semente} | features: {', '.join(feats)}")
    print(r.drop(columns="experimento").to_string(index=False))
    print(f"Figura: {pasta / 'pertinencias.png'}")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="ham10000_todas")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--todos", action="store_true", help="roda para todos os experimentos que tiverem resultados")
    ap.add_argument("--saida", default=None, help="pasta de resultados (padrão: config.SAIDA)")
    a = ap.parse_args()
    if not a.todos:
        desenhar(a.exp, a.semente, a.saida)
    else:
        raiz = Path(a.saida or C.SAIDA); todas = []
        for exp in C.EXPERIMENTOS:
            if (raiz / exp / f"semente_{a.semente}" / "features.csv.gz").exists():
                todas.append(desenhar(exp, a.semente, a.saida))
        pd.concat(todas).to_csv(raiz / "resumo_regras.csv", index=False)
        print(f"\nRegras de todos os experimentos: {raiz / 'resumo_regras.csv'}")
