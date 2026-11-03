"""
exemplo_mnist.py — teste da biblioteca com dados reais que já vêm dentro de um pacote (5.000 dígitos do MNIST
do mlxtend: pip install mlxtend), sem baixar nada. Treina uma CNN e uma MLP, lê as duas com o LeitorFuzzy
e compara com a confiança do softmax.

    python -m leitor_fuzzy.exemplo_mnist
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from leitor_fuzzy import LeitorFuzzy, treinar_rede


def redes():
    cnn = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    return {"CNN": cnn, "MLP": mlp}


def main(semente=0, epocas=4):
    from mlxtend.data import mnist_data
    X, y = mnist_data()
    X = torch.tensor(X / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28); y = torch.tensor(y)
    i = np.random.default_rng(semente).permutation(len(X))
    tr, cal, te = i[:2500], i[2500:3500], i[3500:]                   # treino / calibração / teste
    resultados = {}
    for nome, rede in redes().items():
        torch.manual_seed(semente)
        print(f"\n=== {nome} ===")
        treinar_rede(rede, (X[tr], y[tr]), epocas=epocas)
        for vistas in ("deslocamento", None):                        # dígito espelhado muda de classe: só deslocamento
            lf = LeitorFuzzy(rede, vistas=vistas).fit((X[tr], y[tr]), (X[cal], y[cal]))
            F = lf.caracteristicas((X[te], y[te]))
            r = lf.risco(None, F); e = F.alucinou.values
            auc_f, auc_msp = roc_auc_score(e, r), roc_auc_score(e, 1 - F.confianca)
            tag = "com vistas" if vistas else "sem vistas"
            print(f"  {tag}: acurácia {1 - e.mean():.3f} | alucinações no teste {e.sum()} | "
                  f"AUROC fuzzy {auc_f:.3f} vs softmax {auc_msp:.3f}")
            resultados[(nome, tag)] = (auc_f, auc_msp)
        print(lf.diagnostico((X[te], y[te])).to_string(index=False))
    return resultados


if __name__ == "__main__":
    main()
