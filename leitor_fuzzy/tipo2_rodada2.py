"""Rodada 2 do tipo 2 no MNIST: 5 sementes novas, limiares com meta simples x com garantia (Clopper-Pearson 95%).
    python -m leitor_fuzzy.tipo2_rodada2"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from mlxtend.data import mnist_data
from leitor_fuzzy import LeitorFuzzy, treinar_rede
from leitor_fuzzy.leitor import CARACTERISTICAS
from leitor_fuzzy.tipo2_mnist import redes
from leitor_fuzzy.tipo2_melhorias import VISTAS, avaliar
X, y = mnist_data(); X = torch.tensor(X / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28); y = torch.tensor(y)
cand = [c for c in CARACTERISTICAS if c != "p"] + ["prox_p"]; R = []
for s in (3, 4, 5, 6, 7):
    i = np.random.default_rng(s).permutation(len(X)); tr, te = i[:2500], i[2500:]; tabs = {}
    for nome, rede in redes().items():
        torch.manual_seed(s); treinar_rede(rede, (X[tr], y[tr]), epocas=4, log=None)
        lf = LeitorFuzzy(rede, vistas=VISTAS["giro 10°"]).preparar((X[tr], y[tr]), log=lambda *a: None)
        F = lf.caracteristicas((X[te], y[te])); F["erro"], F["prox_p"] = F.alucinou, 1 - F.confianca; tabs[nome] = F
    for g in (False, True):
        for d in avaliar(tabs, cand, 4, garantia=g):
            R.append({"semente": s, "limiares": "com garantia (95%)" if g else "meta simples", **d})
    print(f"  semente {s} pronta", flush=True)
R = pd.DataFrame(R); pd.set_option("display.width", 250)
cols = ["AUROC", "erro da rede", "seguro %", "erro no seguro", "alarme %", "erro no alarme", "incerto %", "erro no incerto", "erros que escapam", "erros no alarme"]
print("\nMédia (3 redes alvo × 5 sementes novas):"); print(R.groupby("limiares")[cols].mean().round(3).to_string())
print("\nDesvio entre sementes:"); print(R.groupby(["limiares", "semente"])[cols].mean().groupby("limiares").std().round(3).to_string())
print("\nQuantas vezes o erro no SEGURO passou de 3% (de 15):"); print(R.assign(passou=R["erro no seguro"] > 0.03).groupby("limiares").passou.sum().to_string())
print("\nPor rede alvo:"); print(R.groupby(["limiares", "alvo"])[["AUROC", "seguro %", "erro no seguro", "erro no alarme", "incerto %", "erros que escapam"]].mean().round(3).to_string())
R.to_csv("tipo2_rodada2.csv", index=False)
