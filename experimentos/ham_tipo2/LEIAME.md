# Regras fuzzy entre arquiteturas: tipo 1 × tipo 2

Treina EfficientNet-B0, ResNet-50 e ViT-B/16 no HAM10000 (mesmo split por lesão), lê cada rede por dentro
com probes em todos os blocos (profundidade de 0 a 1) e testa se o sistema fuzzy aditivo aprendido em duas redes
funciona na terceira — com termos tipo 1 e com termos tipo 2 (faixa = discordância entre as redes).
Referências: só P, regressão logística e boosting. O Wang-Mendel roda só para consulta (abas `interno_*`).

## Opção A — Kaggle (mais fácil)
1. Suba `tipo2_entre_arquiteturas.ipynb` (File → Import notebook).
2. Add Input → dataset **Skin Cancer MNIST: HAM10000** (kmader).
3. Settings → Accelerator → **GPU**; Settings → **Internet → On**.
4. Confira os caminhos na célula de configuração e clique em **Run All**.
5. Baixe `/kaggle/working/saida_tipo2/resultados_tipo2.xlsx` (e as figuras `fig_*.png`).

Se a sessão cair, rode de novo: checkpoints (`rede_*.pt`), limiares e features ficam em `saida_tipo2` e são reaproveitados.
Para começar do zero, apague a pasta `saida_tipo2`.

## Opção B — qualquer máquina com GPU (terminal)
```bash
pip install -r requirements.txt          # torch com CUDA: siga pytorch.org para a sua versão de CUDA
python teste_rapido.py                   # 3-5 min, imagens falsas: só confere se tudo roda
python rodar.py --metadata /dados/HAM10000_metadata.csv \
                --imagens /dados/HAM10000_images_part_1 /dados/HAM10000_images_part_2 \
                --saida saida_tipo2
```
Sem internet: baixe os pesos do torchvision antes e use `--pesos resnet50=/caminho/resnet50.pth ...`.

## Arquivos
| arquivo | o que faz |
|---|---|
| `redes.py` | dados, split por lesão, as arquiteturas (com os pontos de leitura de cada uma), treino, limiar |
| `leitura_geral.py` | probes em cada bloco de qualquer rede; features em profundidade relativa |
| `fuzzy_t2.py` | fuzzy aditivo e Wang-Mendel, tipo 1 e tipo 2 (envelope), intervalos, zonas, deixa-uma-arquitetura-fora |
| `experimento.py` | o pipeline inteiro (usado pelo notebook e pelo `rodar.py`) |
| `build.py` | regenera o notebook a partir dos módulos |

## O que sai na planilha
- `redes`: AUROC e taxa de erro de cada rede no teste, limiar, nº de pontos de leitura
- `resumo_AUROC`: método × rede deixada de fora
- `deixa_uma_fora`: AUROC, AUROC além de P, AUPR do erro, AURC e o ganho do tipo 2 com IC 95%
- `zonas`: alarme / incerto / seguro (tipo 2) e alarme / seguro (tipo 1), com a taxa de erro em cada zona
- `regras_t2_*` e `faixas_*`: as regras do fuzzy aditivo e onde fica o pico de cada termo em cada rede
- `notas_*`: a nota de cada imagem em cada método (para gráficos)
- `interno_wm_*`: Wang-Mendel, só para consulta (fora do artigo)
