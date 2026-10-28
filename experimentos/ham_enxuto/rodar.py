"""
rodar.py — o mesmo experimento do notebook incerteza_fuzzy_enxuto.ipynb (rodada 5), pelo terminal, sem os gráficos.

    python rodar.py --metadata /dados/HAM10000_metadata.csv \
                    --imagens /dados/HAM10000_images_part_1 /dados/HAM10000_images_part_2 \
                    --modelo /modelos/effnet_ham_best.pt

Saída: incerteza_fuzzy_resultados.xlsx (as mesmas abas do notebook).
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import models, transforms

import leitura_rede as lr
import regras_fuzzy as rf

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Incerteza lida por dentro da EfficientNet do BRACIS + regras fuzzy (HAM10000)")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--imagens", nargs="+", required=True)
    ap.add_argument("--modelo", required=True, help="effnet_ham_best.pt")
    ap.add_argument("--split-csv", default=None, help="split por lesão do notebook de retreino; sem ele = split antigo do BRACIS")
    ap.add_argument("--limiar-json", default=None)
    ap.add_argument("--limiar", type=float, default=0.5)
    ap.add_argument("--n-ref", type=int, default=1000)
    ap.add_argument("--k", type=int, default=4, help="features por árvore / fuzzy aditivo")
    ap.add_argument("--prof", type=int, default=2)
    ap.add_argument("--estagios", action="store_true", help="lê os 9 estágios em vez dos 18 sub-blocos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--saida", default="incerteza_fuzzy_resultados.xlsx")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)

    # dados (igual à seção 2 do notebook)
    meta = pd.read_csv(a.metadata)
    meta["is_malignant"] = meta.dx.isin({"mel", "bcc", "akiec"}).astype(int)
    caminho = {p.stem: str(p) for d in a.imagens for p in Path(d).glob("*.jpg")}
    meta["image_path"] = meta.image_id.map(caminho); meta = meta.dropna(subset=["image_path"])
    limiar = json.load(open(a.limiar_json))["limiar"] if a.limiar_json else a.limiar
    if a.split_csv:
        meta = meta.merge(pd.read_csv(a.split_csv)[["image_id", "split"]], on="image_id")
        treino, cal = meta[meta.split == "train"], meta[meta.split == "test"]
    else:
        m3 = meta.iloc[:3000]
        treino, cal = train_test_split(m3, test_size=0.2, stratify=m3.is_malignant, random_state=42)
        cal = cal[~cal.lesion_id.isin(treino.lesion_id)]
    ref = treino.sample(n=min(a.n_ref, len(treino)), random_state=a.seed)
    cal = cal.reset_index(drop=True)
    print(f"referência: {len(ref)} | calibração: {len(cal)} imagens | limiar: {limiar:.3f}")

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1), nn.Sigmoid())
    model.load_state_dict(torch.load(a.modelo, map_location=dev)); model.to(dev).eval()
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # leitura (seção 3)
    leitor = lr.LeitorRede(model, limiar=limiar, blocos=lr.blocos_efficientnet(model, not a.estagios)).fit(
        lr.lotes_de_imagens(ref.image_path, tfm, dev, a.bs), ref.is_malignant)
    tab = lr.ler(leitor, cal, tfm, dev, a.bs)
    print(f"erros da rede: {tab.erro.sum()} de {len(tab)} ({tab.erro.mean():.1%})")

    # diagnóstico e regras (seções 4 e 5)
    res, escolhas, oof = rf.comparar(tab, lr.FEATURES, k=a.k, prof=a.prof, seed=a.seed)
    diag = rf.diagnostico(tab, lr.FEATURES, escolhas)
    feats = rf.selecionar(tab, lr.FEATURES, a.k)
    arv = rf.ArvoreFuzzy(a.prof).fit(tab[feats].values, tab.erro.values, feats)
    adi = rf.FuzzyAditivo().fit(tab[feats].values, tab.erro.values, feats)
    feats_wm = rf.selecionar(tab, lr.FEATURES, 3)
    wm = rf.WangMendel().fit(tab[feats_wm].values, tab.erro.values, feats_wm)
    est = rf.estabilidade(tab, feats, a.prof)
    print(res.round(3).to_string())
    print(f"features: {feats} | estabilidade das regras da árvore: {est:.2f}")

    # salvar (seção 6)
    resumo = pd.Series({"imagens de calibração": len(tab), "erros da rede": int(tab.erro.sum()), "limiar": limiar,
                        "split": "por lesão (retreino)" if a.split_csv else "antigo (BRACIS)", "features da árvore": ", ".join(feats),
                        "estabilidade das regras": round(est, 3), "blocos lidos": leitor.B, "sub-blocos": not a.estagios,
                        "imagens de referência": len(ref)})
    with pd.ExcelWriter(a.saida) as xw:
        resumo.to_frame("valor").to_excel(xw, sheet_name="resumo")
        diag.round(4).to_excel(xw, sheet_name="diagnostico_features", index=False)
        res.round(4).to_excel(xw, sheet_name="comparacao", index=False)
        arv.regras().to_excel(xw, sheet_name="regras", index=False)
        adi.regras().to_excel(xw, sheet_name="regras_aditivo", index=False)
        wm.regras().to_excel(xw, sheet_name="interno_regras_wang_mendel", index=False)
        tab.to_excel(xw, sheet_name="features_por_imagem", index=False)
    print("Salvo:", a.saida)
