"""
rodar.py — roda o experimento inteiro pelo terminal (qualquer máquina com GPU).

    python rodar.py --metadata /dados/HAM10000_metadata.csv \
                    --imagens /dados/HAM10000_images_part_1 /dados/HAM10000_images_part_2 \
                    --saida saida_tipo2

Se cair no meio, rode o mesmo comando de novo: checkpoints e features em --saida são reaproveitados.
"""
import argparse

import experimento as ex

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regras fuzzy entre arquiteturas: tipo 1 × tipo 2")
    ap.add_argument("--metadata", required=True, help="HAM10000_metadata.csv")
    ap.add_argument("--imagens", nargs="+", required=True, help="pastas com os .jpg")
    ap.add_argument("--saida", default="saida_tipo2")
    ap.add_argument("--arqs", nargs="+", default=["efficientnet_b0", "resnet50", "vit_b_16"],
                    help="efficientnet_b0 resnet50 vit_b_16 convnext_tiny")
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n-ref", type=int, default=2000)
    ap.add_argument("--k", type=int, default=3, help="features por regra do Wang-Mendel")
    ap.add_argument("--k-aditivo", type=int, default=5, help="features do fuzzy aditivo / regressão logística / boosting")
    ap.add_argument("--pesos", nargs="*", default=[], help="pesos locais do ImageNet: nome=caminho.pth")
    ap.add_argument("--max-imagens", type=int, default=0, help=">0: usa só uma amostra de lesões (teste)")
    a = ap.parse_args()
    cfg = ex.Config(metadata_csv=a.metadata, img_dirs=a.imagens, saida=a.saida, arquiteturas=a.arqs,
                    epocas=a.epocas, bs=a.bs, n_ref=a.n_ref, k_features=a.k, k_aditivo=a.k_aditivo, max_imagens=a.max_imagens,
                    pesos_locais=dict(p.split("=", 1) for p in a.pesos))
    r = ex.rodar_tudo(cfg)
    print(r["res"].pivot_table(index="método", columns="rede alvo", values="AUROC").round(3).to_string())
