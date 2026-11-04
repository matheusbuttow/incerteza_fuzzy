"""
teste_tipos.py — a biblioteca com tipo 1 × tipo 2, no MNIST real (5.000 dígitos do mlxtend), cada rede lida por ela mesma:
treino 2.500 (rede, sondas, vizinhos) · calibração 1.000 (regras e limiares das zonas) · teste 1.500.

    python -m leitor_fuzzy.teste_tipos
"""
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score

from leitor_fuzzy import LeitorFuzzy, treinar_rede
from leitor_fuzzy.tipo2_mnist import redes


def main(sementes=(3, 4, 5, 6, 7), epocas=4):
    from mlxtend.data import mnist_data
    X, y = mnist_data()
    X = torch.tensor(X / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28); y = torch.tensor(y)
    giro = [lambda x: TF.rotate(x, 10.0)]
    R = []
    for s in sementes:
        i = np.random.default_rng(s).permutation(len(X)); tr, cal, te = i[:2500], i[2500:3500], i[3500:]
        for nome, rede in redes().items():
            torch.manual_seed(s); treinar_rede(rede, (X[tr], y[tr]), epocas=epocas, log=None)
            base = LeitorFuzzy(rede, vistas=giro).preparar((X[tr], y[tr]), log=lambda *a: None)
            Fc, Ft = base.caracteristicas((X[cal], y[cal])), base.caracteristicas((X[te], y[te]))
            for tipo in (1, 2):
                lf = LeitorFuzzy(rede, vistas=giro, tipo=tipo)
                lf.__dict__.update({k: base.__dict__[k] for k in ("blocos", "cabeca", "sondas", "banco", "banco_dec", "dist_ref", "n_classes")})
                lf.ajustar(Fc, log=lambda *a: None)
                Z = lf.classificar(None, Ft); e = Z.alucinou.values
                z = {nz: (Z.zona == nz).values for nz in ("seguro", "incerto", "alarme")}
                R.append({"semente": s, "rede": nome, "tipo": f"tipo {tipo}", "AUROC": roc_auc_score(e, Z.risco),
                          "erro da rede": e.mean(), "seguro %": z["seguro"].mean(),
                          "erro no seguro": e[z["seguro"]].mean() if z["seguro"].any() else np.nan,
                          "alarme %": z["alarme"].mean(), "erro no alarme": e[z["alarme"]].mean() if z["alarme"].any() else np.nan,
                          "incerto %": z["incerto"].mean(), "erros avisados": e[~z["seguro"]].sum() / e.sum(),
                          "erros que escapam": e[z["seguro"]].sum() / e.sum(), "meta seguro": lf.meta_seguro_usada,
                          "largura": (Z.risco_sup - Z.risco_inf).mean()})
        print(f"  semente {s} pronta", flush=True)
    R = pd.DataFrame(R); pd.set_option("display.width", 250)
    cols = ["AUROC", "erro da rede", "seguro %", "erro no seguro", "meta seguro", "alarme %", "erro no alarme",
            "incerto %", "erros avisados", "erros que escapam", "largura"]
    print(f"\nMédia (3 redes × {len(sementes)} sementes):"); print(R.groupby("tipo")[cols].mean().round(3).T.to_string())
    print("\nVezes em que o erro no seguro passou da meta:")
    print(R.assign(p=R["erro no seguro"] > R["meta seguro"]).groupby("tipo").p.sum().to_string())
    P = R.pivot_table(index=["semente", "rede"], columns="tipo", values=["erros que escapam", "seguro %"])
    for c in ("erros que escapam", "seguro %"):
        d = P[c]["tipo 2"] - P[c]["tipo 1"]
        print(f"  {c}: tipo 2 − tipo 1 = {d.mean():+.4f} (tipo 2 {'menor' if c == 'erros que escapam' else 'maior'} em "
              f"{(d < 0).sum() if c == 'erros que escapam' else (d > 0).sum()} de {len(d)})")
    print("\nPor rede:"); print(R.groupby(["rede", "tipo"])[["AUROC", "seguro %", "erro no seguro", "incerto %", "erros que escapam"]].mean().round(3).to_string())
    return R


if __name__ == "__main__":
    main().to_csv("teste_tipos.csv", index=False)
