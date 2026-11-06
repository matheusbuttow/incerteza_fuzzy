"""
analise_ruido.py — a curva de ruído: o ganho do nosso sistema cresce com a fração de rótulos de treino trocados?
Junta cifar10 (0%), cifar10_ruido10/20/30/40 (mesma ResNet-32) e mede, em cada nível:
  · AUROC em todas as imagens (softmax, Só P, Chen, nosso fuzzy)
  · AUROC dentro das 25% imagens em que a rede está mais confiante (o erro confiante), com o fuzzy só dos confiantes

    python analise_ruido.py                      # → resultados/ruido.xlsx e resultados/figuras/ruido.png
Também junta o ruído REAL (CIFAR-10N / CIFAR-100N, rótulos dados por pessoas), numa aba à parte ("humano").
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import analise_confiantes as ac

NIVEIS = {"cifar10": 0, "cifar10_ruido10": 10, "cifar10_ruido20": 20, "cifar10_ruido30": 30, "cifar10_ruido40": 40}
# ruído REAL (rótulos de pessoas, CIFAR-N): entra numa tabela à parte, com a taxa de erro real dos rótulos
HUMANOS = {"cifar10n_aggre": "CIFAR-10N voto de 3 (~9%)", "cifar10n_random1": "CIFAR-10N 1 pessoa (~17%)",
           "cifar10n_worst": "CIFAR-10N pior de 3 (~40%)", "cifar100n": "CIFAR-100N 1 pessoa (~40%)",
           "cifar100_vgg16": "CIFAR-100 sem ruído (referência)"}
METODOS = ["Softmax (MSP)", "Só P", "Chen et al. 2019", "NOSSO: Fuzzy aditivo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default="resultados")
    ap.add_argument("--frac", type=float, default=0.25, help="fração mais confiante para o erro confiante")
    a = ap.parse_args()
    pasta = Path(a.pasta); L = []
    for exp, nivel in list(NIVEIS.items()) + list(HUMANOS.items()):
        for p in sorted((pasta / exp).glob("semente_*/escores.csv.gz")):
            sem = int(p.parent.name.split("_")[1])
            F = pd.read_csv(p.parent / "features.csv.gz"); E = pd.read_csv(p); y = F.erro.values
            lin = {"ruído (%)": nivel, "semente": sem, "erro da rede": y.mean(), "humano": exp in HUMANOS}
            for m in METODOS:
                lin[f"todas: {m}"] = roc_auc_score(y, E[m])
            corte = np.quantile(F.prox_p, a.frac); g = (F.prox_p <= corte).values
            G, yg = F[g].reset_index(drop=True), y[g]
            lin["confiantes: erros no grupo"] = int(yg.sum())
            if yg.sum() >= 5:
                for m in METODOS:
                    lin[f"confiantes: {m}"] = roc_auc_score(yg, E[m].values[g])
                s, _ = ac.fuzzy_no_grupo(G, [c for c in ac.INTERNAS if c in G], 3, False, sem); ok = ~np.isnan(s)
                if ok.any() and len(np.unique(yg[ok])) == 2:
                    lin["confiantes: NOSSO: fuzzy só dos confiantes"] = roc_auc_score(yg[ok], s[ok])
            L.append(lin)
            print(f"  ruído {nivel}{'' if exp in HUMANOS else '%'} semente {sem}: erro {y.mean():.3f} | nosso {lin['todas: NOSSO: Fuzzy aditivo']:.3f} "
                  f"× softmax {lin['todas: Softmax (MSP)']:.3f}", flush=True)
    if not L:
        raise SystemExit("Nenhuma rodada de cifar10 / cifar10_ruido* encontrada.")
    L = pd.DataFrame(L); H = L[L.humano].drop(columns="humano"); L = L[~L.humano].drop(columns="humano")
    if len(H):
        RH = H.groupby("ruído (%)", sort=False).mean(numeric_only=True).drop(columns="semente")
        RH["ganho sobre softmax (todas)"] = RH["todas: NOSSO: Fuzzy aditivo"] - RH["todas: Softmax (MSP)"]
        RH["ganho sobre Chen (todas)"] = RH["todas: NOSSO: Fuzzy aditivo"] - RH["todas: Chen et al. 2019"]
        print("\nRUÍDO HUMANO (CIFAR-N):\n" + RH.round(3).T.to_string())
    if not len(L):
        raise SystemExit("Nenhuma rodada da curva sintética.")
    R = L.groupby("ruído (%)").mean(numeric_only=True).drop(columns="semente")
    R["ganho sobre softmax (todas)"] = R["todas: NOSSO: Fuzzy aditivo"] - R["todas: Softmax (MSP)"]
    R["ganho sobre Chen (todas)"] = R["todas: NOSSO: Fuzzy aditivo"] - R["todas: Chen et al. 2019"]
    pd.set_option("display.width", 250)
    print("\n" + R.round(3).T.to_string())
    with pd.ExcelWriter(pasta / "ruido.xlsx") as xw:
        R.round(4).to_excel(xw, sheet_name="curva")
        L.round(4).to_excel(xw, sheet_name="por_semente", index=False)
        if len(H):
            RH.round(4).to_excel(xw, sheet_name="humano")
            H.round(4).to_excel(xw, sheet_name="humano_por_semente", index=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        estilo = {"Softmax (MSP)": ("#888888", "--"), "Só P": ("#bbbbbb", ":"), "Chen et al. 2019": ("#d98c1f", "-"),
                  "NOSSO: Fuzzy aditivo": ("#1f5fa8", "-"), "NOSSO: fuzzy só dos confiantes": ("#1f5fa8", "--")}
        for i, (pref, titulo) in enumerate([("todas", "todas as imagens"), ("confiantes", f"{int(100 * a.frac)}% mais confiantes")]):
            for m, (cor, ls) in estilo.items():
                col = f"{pref}: {m}"
                if col in R:
                    ax[i].plot(R.index, R[col], ls, color=cor, marker="o", label=m.replace("NOSSO: ", "nosso: "))
            ax[i].set_xlabel("rótulos de treino trocados (%)"); ax[i].set_ylabel("AUROC (detectar o erro)"); ax[i].set_title(titulo)
            ax[i].grid(alpha=.3)
        ax[1].axhline(0.5, color="k", lw=.6); ax[1].legend(fontsize=8, loc="lower left")
        (pasta / "figuras").mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(pasta / "figuras" / "ruido.png", dpi=150)
        print(f"Figura: {pasta / 'figuras' / 'ruido.png'}")
    except Exception as err:
        print("(figura não gerada:", err, ")")
    print(f"Planilha: {pasta / 'ruido.xlsx'}")


if __name__ == "__main__":
    main()
