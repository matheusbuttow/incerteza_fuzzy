"""
conferir_resultados.py — procura rodadas com arquivos corrompidos em resultados/ (ex.: duas cópias do script escrevendo o mesmo
arquivo ao mesmo tempo, ou o PC desligado no meio de uma gravação) e downloads de datasets quebrados em dados/.

    python conferir_resultados.py            # só lista o que está estragado
    python conferir_resultados.py --mover    # move as rodadas estragadas para resultados_corrompidos/ (nada é apagado)

Depois de mover, rode de novo o rodar_3dias.sh: só as rodadas que faltam são refeitas.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

ARQUIVOS = {"features.csv.gz": lambda p: pd.read_csv(p), "escores.csv.gz": lambda p: pd.read_csv(p),
            "zonas_por_imagem.csv.gz": lambda p: pd.read_csv(p), "metricas.csv": lambda p: pd.read_csv(p),
            "extras.npz": lambda p: dict(np.load(p)), "modelo.pt": None}


def conferir_rodada(d):
    """Lista de problemas da pasta de uma rodada (vazia = tudo certo)."""
    ruins = []
    for nome, ler in ARQUIVOS.items():
        p = d / nome
        if not p.exists():
            continue
        try:
            if nome == "modelo.pt":
                import torch
                torch.load(p, map_location="cpu")
            else:
                ler(p)
        except Exception as err:
            ruins.append(f"{nome}: {type(err).__name__}: {str(err)[:80]}")
    return ruins


def conferir_dados():
    """Tenta abrir os datasets do torchvision já baixados; um download quebrado aparece aqui."""
    from torchvision import datasets
    out = []
    for nome, cls, pasta in [("MNIST", datasets.MNIST, "MNIST"), ("Fashion-MNIST", datasets.FashionMNIST, "FashionMNIST")]:
        if (C.DADOS / pasta).exists():
            try:
                cls(str(C.DADOS), train=True, download=False); cls(str(C.DADOS), train=False, download=False)
            except Exception as err:
                out.append((nome, C.DADOS / pasta, f"{type(err).__name__}: {str(err)[:80]}"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", default=str(C.SAIDA))
    ap.add_argument("--mover", action="store_true", help="move as rodadas e downloads estragados para pastas *_corrompidos")
    a = ap.parse_args()
    pasta = Path(a.pasta); destino = Path(str(pasta) + "_corrompidos")
    rodadas = sorted(d for d in pasta.glob("*/semente_*") if d.is_dir())
    ruins = {d: r for d in rodadas if (r := conferir_rodada(d))}
    print(f"{len(rodadas)} rodadas conferidas em {pasta}/: {len(ruins)} com arquivo estragado")
    for d, r in ruins.items():
        print(f"  {d}"); [print(f"      {x}") for x in r]
    dados = conferir_dados()
    for nome, p, err in dados:
        print(f"  download estragado: {nome} ({p}) — {err}")
    if not ruins and not dados:
        print("Tudo certo."); return
    if not a.mover:
        print("\nPara mover o que está estragado (nada é apagado) e refazer só isso:\n"
              "  python conferir_resultados.py --mover\n  nohup bash rodar_3dias.sh > log_3dias.txt 2>&1 &"); return
    for d in ruins:
        alvo = destino / d.parent.name / d.name
        alvo.parent.mkdir(parents=True, exist_ok=True)
        if alvo.exists():
            shutil.rmtree(alvo)
        shutil.move(str(d), str(alvo)); print(f"  movida: {d} -> {alvo}")
    for nome, p, _ in dados:
        alvo = p.parent / (p.name + "_corrompido")
        if alvo.exists():
            shutil.rmtree(alvo)
        shutil.move(str(p), str(alvo)); print(f"  movido: {p} -> {alvo} (será baixado de novo)")
    print("\nAgora rode de novo: nohup bash rodar_3dias.sh > log_3dias.txt 2>&1 &   (confira antes: pgrep -af rodar)")


if __name__ == "__main__":
    main()
