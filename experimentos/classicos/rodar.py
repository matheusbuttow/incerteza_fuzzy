"""
rodar.py — o mesmo experimento do notebook classicos_comparacao.ipynb, pelo terminal.

    python rodar.py                                   # CIFAR-10, CIFAR-100 e Tiny-ImageNet, limpo e ruidoso
    python rodar.py --datasets cifar10 --cenarios ruidoso
    python rodar.py --tiny-dir /dados/tiny-imagenet-200   # sem baixar o Tiny-ImageNet

Os modelos treinados ficam em --modelos; rodar de novo pula o treino. Saída: classicos_resultados.xlsx.
"""
import argparse

import classicos as cc

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Protocolo de Chen et al. 2019 em datasets clássicos + nosso método")
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100", "tinyimagenet"], choices=list(cc.DATASETS))
    ap.add_argument("--cenarios", nargs="+", default=["limpo", "ruidoso"], choices=["limpo", "ruidoso"])
    ap.add_argument("--modelo", default="chen", choices=["chen", "nosso"])
    ap.add_argument("--passos", type=int, default=90_000, help="passos de treino (modelo chen)")
    ap.add_argument("--epocas", type=int, default=60, help="épocas (modelo nosso)")
    ap.add_argument("--k", type=int, default=4, help="features dos sistemas fuzzy")
    ap.add_argument("--sem-ood", action="store_true")
    ap.add_argument("--dados", default="./dados")
    ap.add_argument("--tiny-dir", default=None)
    ap.add_argument("--modelos", default=".", help="pasta dos .pt treinados")
    ap.add_argument("--saida", default="classicos_resultados.xlsx")
    a = ap.parse_args()
    frac = {"limpo": 0.0, "ruidoso": 0.3}
    saidas, res, prot = cc.rodar_tudo(a.datasets, [(c, frac[c]) for c in a.cenarios], a.modelo, a.passos, a.epocas, a.k,
                                      not a.sem_ood, a.dados, a.tiny_dir, a.modelos)
    print(prot.round(3).to_string())
    art = res[(res.uso == "artigo") & (res.conjunto == "TEST")]
    print(art.pivot_table(index="método", columns=["dataset", "cenário"], values="AUROC", sort=False).round(3).to_string())
    print("Salvo:", cc.salvar(saidas, res, prot, a.saida))
