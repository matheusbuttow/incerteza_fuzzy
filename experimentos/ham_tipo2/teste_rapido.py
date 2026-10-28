"""
teste_rapido.py — confere se tudo roda, com imagens FALSAS (manchas aleatórias), em poucos minutos.
Não serve como resultado: as redes não aprendem nada de verdade aqui.

    python teste_rapido.py                     # CPU ou GPU
    python teste_rapido.py --arqs resnet50     # só uma arquitetura... (o deixa-uma-fora precisa de pelo menos 2)
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import experimento as ex


def dados_falsos(pasta, n_lesoes=90, seed=0):
    r = np.random.default_rng(seed); pasta = Path(pasta); (pasta / "imgs").mkdir(parents=True, exist_ok=True)
    linhas = []
    for l in range(n_lesoes):
        mal = int(r.random() < 0.3)
        for v in range(r.integers(1, 3)):
            iid = f"ISIC_{l:04d}{v}"
            im = Image.new("RGB", (120, 90), tuple(int(c) for c in r.integers(150, 230, 3)))
            d = ImageDraw.Draw(im); cx, cy, raio = r.integers(40, 80), r.integers(30, 60), r.integers(10, 30)
            cor = (90, 40, 30) if mal else (140, 100, 80)
            d.ellipse([cx - raio, cy - raio * (1.5 if mal else 1), cx + raio, cy + raio], fill=cor)
            im.save(pasta / "imgs" / f"{iid}.jpg")
            linhas.append({"lesion_id": f"HAM_{l:04d}", "image_id": iid,
                           "dx": r.choice(["mel", "bcc"]) if mal else r.choice(["nv", "bkl"])})
    pd.DataFrame(linhas).to_csv(pasta / "meta.csv", index=False)
    return pasta / "meta.csv", [str(pasta / "imgs")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arqs", nargs="+", default=["efficientnet_b0", "resnet50", "vit_b_16"])
    a = ap.parse_args()
    shutil.rmtree("teste_falso", ignore_errors=True)
    meta_csv, dirs = dados_falsos("teste_falso/dados")
    cfg = ex.Config(metadata_csv=str(meta_csv), img_dirs=dirs, saida="teste_falso/saida", arquiteturas=a.arqs,
                    epocas=1, bs=16, lado_cache=64, n_ref=60, frac_teste=0.4, n_boot=50, pretreinado=False)
    r = ex.rodar_tudo(cfg)
    print(r["res"].pivot_table(index="método", columns="rede alvo", values="AUROC").round(3))
    print("OK — tudo rodou. Planilha:", r["planilha"])
