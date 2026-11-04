"""
tipo2_melhorias.py — três tentativas de melhorar o tipo 2 (MNIST real, 3 redes, regras aprendidas nas OUTRAS duas):
  1. limiares com meta: SEGURO só onde o erro (nas redes de origem) é ≤ META_SEGURO; ALARME onde é ≥ META_ALARME
  2. transformação mais forte: giro de 10° em vez de deslocar 1 pixel (2 passadas nos dois casos)
  3. mais características: 4, 5 ou 6

    python -m leitor_fuzzy.tipo2_melhorias
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "experimentos" / "ham_tipo2"))
import fuzzy_t2 as t2  # noqa: E402

from leitor_fuzzy import LeitorFuzzy, treinar_rede  # noqa: E402
from leitor_fuzzy.leitor import CARACTERISTICAS  # noqa: E402
from leitor_fuzzy.tipo2_mnist import redes  # noqa: E402

META_SEGURO, META_ALARME = 0.03, 0.50
VISTAS = {"desloca 1px": [lambda x: torch.roll(x, (1, 1), (-2, -1))],
          "giro 10°": [lambda x: TF.rotate(x, 10.0)]}


def _taxa(erros, n, garantia, lado):
    """Taxa de erro observada, ou o limite de Clopper–Pearson (95%) quando garantia=True:
    'sup' para o seguro (pior caso = mais erro), 'inf' para o alarme (pior caso = menos erro)."""
    if not garantia:
        return erros / n
    from scipy.stats import beta
    return beta.ppf(0.95, erros + 1, n - erros) if lado == "sup" else beta.ppf(0.05, np.maximum(erros, 1e-9), n - erros + 1)


def limiar_seguro(s_sup, y, garantia=False):
    """Maior t tal que, entre as imagens com s_sup < t, a taxa de erro (ou o seu limite superior) fica ≤ META_SEGURO."""
    o = np.argsort(s_sup); n = np.arange(1, len(y) + 1)
    ok = np.where(_taxa(np.cumsum(y[o]), n, garantia, "sup") <= META_SEGURO)[0]
    return s_sup[o][ok[-1]] if len(ok) else -np.inf


def limiar_alarme(s_inf, y, garantia=False):
    """Menor t tal que, entre as imagens com s_inf > t, a taxa de erro (ou o seu limite inferior) fica ≥ META_ALARME."""
    o = np.argsort(-s_inf); n = np.arange(1, len(y) + 1)
    ok = np.where(_taxa(np.cumsum(y[o]), n, garantia, "inf") >= META_ALARME)[0]
    return s_inf[o][ok[-1]] if len(ok) else np.inf


def avaliar(tabs, cand, k, garantia=False):
    lin = []
    for alvo in tabs:
        fontes = [s for s in tabs if s != alvo]
        src = pd.concat([tabs[s].assign(_g=i) for i, s in enumerate(fontes)], ignore_index=True)
        feats = t2.selecionar(src, cand, k, ("prox_p",))
        lista_C = [t2.centros(src[src._g == i][feats].values) for i in range(len(fontes))]
        Mi, Ms = t2.pert_t2(src[feats].values, lista_C)
        m = t2.FuzzyAditivo().fit((Mi + Ms) / 2, src.erro.values, feats, src._g.values)
        si_src, ss_src, _ = m.nota_t2(Mi, Ms)
        ts, ta = limiar_seguro(ss_src, src.erro.values, garantia), limiar_alarme(si_src, src.erro.values, garantia)
        T = tabs[alvo]; y = T.erro.values
        si, ss, sm = m.nota_t2(*t2.pert_t2(T[feats].values, lista_C))
        seg, al = ss < ts, (si > ta) & ~(ss < ts); inc = ~seg & ~al
        lin.append({"alvo": alvo, "AUROC": roc_auc_score(y, sm), "erro da rede": y.mean(),
                    "seguro %": seg.mean(), "erro no seguro": y[seg].mean() if seg.any() else np.nan,
                    "alarme %": al.mean(), "erro no alarme": y[al].mean() if al.any() else np.nan,
                    "incerto %": inc.mean(), "erro no incerto": y[inc].mean() if inc.any() else np.nan,
                    "erros que escapam": y[seg].sum() / y.sum(), "erros no alarme": y[al].sum() / y.sum(),
                    "features": ", ".join(feats)})
    return lin


def main(sementes=(0, 1, 2), epocas=4):
    from mlxtend.data import mnist_data
    X, y = mnist_data()
    X = torch.tensor(X / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28); y = torch.tensor(y)
    cand = [c for c in CARACTERISTICAS if c != "p"] + ["prox_p"]
    R = []
    for s in sementes:
        i = np.random.default_rng(s).permutation(len(X)); tr, te = i[:2500], i[2500:]
        nets = {}
        for nome, rede in redes().items():
            torch.manual_seed(s); nets[nome] = treinar_rede(rede, (X[tr], y[tr]), epocas=epocas, log=None)
        for vn, vf in VISTAS.items():
            tabs = {}
            for nome, rede in nets.items():
                lf = LeitorFuzzy(rede, vistas=vf).preparar((X[tr], y[tr]), log=lambda *a: None)
                F = lf.caracteristicas((X[te], y[te]))
                F["erro"], F["prox_p"] = F.alucinou, 1 - F.confianca
                tabs[nome] = F
            for k in (4, 5, 6):
                for d in avaliar(tabs, cand, k):
                    R.append({"semente": s, "vista": vn, "k": k, **d})
        print(f"  semente {s} pronta", flush=True)
    R = pd.DataFrame(R)
    pd.set_option("display.width", 250)
    cols = ["AUROC", "seguro %", "erro no seguro", "alarme %", "erro no alarme", "incerto %", "erros que escapam", "erros no alarme"]
    print(f"\nMetas: erro no SEGURO ≤ {META_SEGURO:.0%}, erro no ALARME ≥ {META_ALARME:.0%} (escolhidas nas redes de origem)")
    print("Média das 3 redes alvo e", len(sementes), "sementes:")
    print(R.groupby(["vista", "k"])[cols].mean().round(3).to_string())
    print("\nPor rede alvo (giro 10°, k=5):")
    print(R[(R.vista == "giro 10°") & (R.k == 5)].groupby("alvo")[cols].mean().round(3).to_string())
    print("\nFeatures escolhidas (giro 10°, k=5):", R[(R.vista == "giro 10°") & (R.k == 5)].features.value_counts().head(3).to_dict())
    return R


if __name__ == "__main__":
    main()
