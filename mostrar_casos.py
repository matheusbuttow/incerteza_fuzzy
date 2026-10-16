"""
mostrar_casos.py — mostra, imagem por imagem, o que o sistema responde: o que a rede disse, se errou,
o risco de erro dado pelo fuzzy aditivo e quanto cada regra pesou. Usa os resultados já salvos
(não treina nem relê a rede).

Exemplos:
  python mostrar_casos.py                                   # 10 primeiras imagens do HAM completo, semente 42
  python mostrar_casos.py --n 20 --semente 7
  python mostrar_casos.py --escolher                        # em vez das primeiras: 3 acertos tranquilos,
                                                            # 3 erros pegos, 3 erros que escaparam, 1 alarme falso
  python mostrar_casos.py --exp cifar10_ruido30 --semente 42

Saída: tabela no terminal + resultados/<exp>/semente_<s>/casos.csv e casos.png (fotos, quando houver caminho).
"""
import argparse
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import fuzzy as fz
from leitura import NOSSAS

warnings.filterwarnings("ignore")
METODO = "NOSSO: Fuzzy aditivo"
NOMES = {"p": "P (resposta da rede)", "salto_final": "salto final", "trajetoria_vistas": "trajetória nas vistas",
         "p_vistas_std": "P muda nas vistas", "vistas_mudam": "decisão muda nas vistas", "conf_fim": "concordância no fim",
         "entropia_probes": "hesitação nas camadas", "trocas_opiniao": "trocas de opinião", "prof_decisao": "decide tarde",
         "dist_vizinhos": "distância dos vizinhos", "vizinhos_discordam": "vizinhos discordam"}


def explicar(tab, fixos):
    """Ajusta o fuzzy aditivo final (o das regras do relatório) e devolve risco e contribuição de cada feature por imagem."""
    cand = [c for c in NOSSAS if c in tab]
    feats = fz.selecionar(tab, cand, C.FEATURES_ADITIVO)
    ad = fz.FuzzyAditivo(fixos).fit(tab[feats].values, tab.erro.values, feats)
    M = ad.part.pertinencias(tab[feats].values); w = ad.lr.coef_[0]; b = ad.lr.intercept_[0]
    contrib = pd.DataFrame({f: (M[j] * w[3 * j: 3 * j + 3]).sum(1) for j, f in enumerate(feats)})
    termo = pd.DataFrame({f: [fz.TERMOS[k] for k in M[j].argmax(1)] for j, f in enumerate(feats)})
    risco = 1 / (1 + np.exp(-(b + contrib.sum(1))))
    return feats, contrib, termo, risco, b


def escolher(tab, risco, n_cada=3):
    lim = np.quantile(risco, 0.8)                                   # "encaminhado" = 20% mais incertos
    enc = risco >= lim
    grupos = [("acerto tranquilo", (tab.erro == 0) & ~enc, risco, True),
              ("erro pego (encaminhado)", (tab.erro == 1) & enc, risco, False),
              ("erro que escapou", (tab.erro == 1) & ~enc, risco, True),
              ("alarme falso (acerto encaminhado)", (tab.erro == 0) & enc, risco, False)]
    idx, rot = [], []
    for nome, m, r, asc in grupos:
        k = 1 if nome.startswith("alarme") else n_cada
        sel = r[m].sort_values(ascending=asc).index[:k]
        idx += list(sel); rot += [nome] * len(sel)
    return idx, rot, lim


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="ham10000_todas")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--escolher", action="store_true")
    ap.add_argument("--saida", default=None, help="pasta de resultados (padrão: config.SAIDA)")
    a = ap.parse_args()
    from pathlib import Path
    pasta = Path(a.saida or C.SAIDA) / a.exp / f"semente_{a.semente}"
    tab = pd.read_csv(pasta / "features.csv.gz")
    info = json.loads((pasta / "info_rede.json").read_text())
    binario = info.get("limiar") is not None
    fixos = {"p": [0.0, info["limiar"], 1.0]} if binario else None
    feats, contrib, termo, risco, base = explicar(tab, fixos)
    risco = pd.Series(risco, index=tab.index)

    if a.escolher:
        idx, rotulo, lim = escolher(tab, risco)
    else:
        idx, rotulo = list(tab.index[: a.n]), [""] * min(a.n, len(tab))
        lim = np.quantile(risco, 0.8)

    classe = (lambda v: "maligna" if v == 1 else "benigna") if binario else (lambda v: str(int(v)))
    lin = []
    for i, r in zip(idx, rotulo):
        t = tab.loc[i]; c = contrib.loc[i]
        principais = c.reindex(c.abs().sort_values(ascending=False).index)[:3]
        lin.append({"caso": r, "imagem": t.get("image_id", i), "real": classe(t.rotulo), "rede disse": classe(t.decisao),
                    "P": round(float(t.p), 3), "rede errou?": "SIM" if t.erro else "não",
                    "risco de erro (aditivo)": f"{100 * risco[i]:.0f}%",
                    "vai ao médico? (20% mais incertos)": "sim" if risco[i] >= lim else "não",
                    **{f"{NOMES.get(f, f)}": f"{termo.loc[i, f]} ({contrib.loc[i, f]:+.2f})" for f in feats},
                    "motivo principal": "; ".join(f"{NOMES.get(f, f)} {termo.loc[i, f]} ({v:+.2f})" for f, v in principais.items())})
    out = pd.DataFrame(lin)
    out.to_csv(pasta / "casos.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 80)
    print(f"\n{a.exp} | semente {a.semente} | {len(tab)} imagens de teste | rede errou em {int(tab.erro.sum())}"
          f" | features do aditivo: {', '.join(feats)} | base {base:+.2f}")
    print(f"'vai ao médico' = risco acima de {100 * lim:.0f}% (os 20% mais incertos)\n")
    for _, r in out.iterrows():
        print(f"[{r['caso'] or r['imagem']}]  {r['imagem']}: real {r['real']}, rede disse {r['rede disse']} (P={r['P']})"
              f" -> rede errou? {r['rede errou?']} | risco {r['risco de erro (aditivo)']} | vai ao médico? {r['vai ao médico? (20% mais incertos)']}")
        print(f"      por quê: {r['motivo principal']}")
    print(f"\nTabela completa: {pasta / 'casos.csv'}")

    if "image_path" in tab:
        from PIL import Image
        n = len(idx); fig, axs = plt.subplots(n, 2, figsize=(10, 2.4 * n), gridspec_kw={"width_ratios": [1, 1.8]})
        axs = np.atleast_2d(axs)
        for (a1, a2), i, r in zip(axs, idx, rotulo):
            t = tab.loc[i]
            try:
                a1.imshow(Image.open(t.image_path).convert("RGB"))
            except Exception:
                a1.text(0.5, 0.5, "foto não encontrada", ha="center")
            a1.axis("off")
            cor = "#b8325a" if t.erro else "#2a78d6"
            a1.set_title(f"{r + ' | ' if r else ''}{t.get('image_id', i)}\nreal {classe(t.rotulo)} | rede {classe(t.decisao)} (P={t.p:.2f})",
                         fontsize=8, loc="left", color=cor)
            c = contrib.loc[i]
            nomes = [f"{NOMES.get(f, f)} {termo.loc[i, f]}" for f in feats]
            a2.barh(nomes, c.values, color=["#eb6834" if v > 0 else "#2a78d6" for v in c.values])
            a2.axvline(0, color="#888", lw=0.8); a2.tick_params(labelsize=7)
            a2.set_title(f"risco de erro {100 * risco[i]:.0f}% | {'vai ao médico' if risco[i] >= lim else 'rede decide sozinha'}"
                         f" | rede errou? {'SIM' if t.erro else 'não'}", fontsize=8, loc="left")
            a2.spines[["top", "right"]].set_visible(False)
        axs[-1, 1].set_xlabel("pontos de risco (laranja = aumenta, azul = diminui)", fontsize=8)
        fig.tight_layout(); fig.savefig(pasta / "casos.png", dpi=130, bbox_inches="tight")
        print(f"Figura: {pasta / 'casos.png'}")
