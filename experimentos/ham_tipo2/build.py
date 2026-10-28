"""Gera tipo2_entre_arquiteturas.ipynb (módulos embutidos via %%writefile)."""
import nbformat as nbf
from pathlib import Path

AQUI = Path(__file__).parent
nb = nbf.v4.new_notebook(); C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Regras fuzzy que valem para qualquer rede: tipo 1 × tipo 2

**Pergunta:** as regras que detectam erro, aprendidas olhando *algumas* redes, funcionam numa rede que o sistema nunca viu?

1. Treina **três redes bem diferentes** no HAM10000, no mesmo split por lesão: EfficientNet-B0, ResNet-50 e ViT-B/16.
2. Lê cada rede por dentro (probes em cada bloco) com a **profundidade medida de 0 a 1**, para as features serem comparáveis.
3. **Deixa uma rede de fora:** aprende os sistemas fuzzy nas outras duas e aplica na que ficou de fora.
   Sistema fuzzy principal: **fuzzy aditivo** (regras de 1 condição somadas, 5 features), que nas rodadas anteriores
   ficou colado na regressão logística. Referências: só P, regressão logística e boosting.
   (O Wang-Mendel também roda, mas só para consulta: aparece no fim do notebook e em abas `interno_*`.)
   - **Tipo 1:** uma curva por termo ("taxa_trocas ALTA"), das redes de origem juntas.
   - **Tipo 2:** a curva vira uma faixa = a discordância entre as redes de origem. A saída vira um intervalo e três zonas: alarme, incerto, seguro.

**Antes de rodar:** Settings → Accelerator → **GPU**; Settings → **Internet → On** (para baixar os pesos do ImageNet).
Tempo estimado numa T4/P100: cerca de 1 a 2 horas para as três redes (o ViT é o mais lento).
Se a sessão cair, rode de novo: checkpoints e features ficam em `/kaggle/working/saida_tipo2` e são reaproveitados.

**Mande o arquivo `saida_tipo2/resultados_tipo2.xlsx`** (e, se quiser, as três figuras `fig_*.png`).
""")
for nome in ["redes.py", "leitura_geral.py", "fuzzy_t2.py", "experimento.py"]:
    code(f"%%writefile {nome}\n" + (AQUI / nome).read_text())

md("## Configuração")
code("""
import experimento as ex
cfg = ex.Config(
    metadata_csv='/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_metadata.csv',
    img_dirs=['/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_1',
              '/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_2'],
    saida='/kaggle/working/saida_tipo2',
    arquiteturas=['efficientnet_b0', 'resnet50', 'vit_b_16'],   # opcional: 'convnext_tiny'
    epocas=15, bs=32,
    pesos_locais={},   # sem internet: {'resnet50': '/kaggle/input/.../resnet50-11ad3fa6.pth', ...}
)
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NENHUMA — ligue a GPU em Settings → Accelerator')
""")

md("## 1. Dados: split por lesão (a mesma lesão nunca aparece em dois conjuntos)")
code("""
import numpy as np, pandas as pd, warnings
from IPython.display import display, Image
warnings.filterwarnings('ignore'); pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
meta, cache = ex.preparar_dados(cfg)
""")

md("""
## 2. Treinar cada rede e ler por dentro

Para cada rede: treino (ou checkpoint), limiar escolhido na validação, e leitura do teste com probes em todos os blocos.
""")
code("""
import fuzzy_t2 as ft, leitura_geral as lg
infos, hists, tabs = [], {}, {}
for nome in cfg.arquiteturas:
    rede, info, hist = ex.treinar_ou_carregar(cfg, nome, meta, cache)
    tabs[nome] = ex.ler_ou_carregar(cfg, nome, rede, info, meta, cache)
    infos.append(info); hists[nome] = hist
    del rede; torch.cuda.empty_cache() if torch.cuda.is_available() else None
display(pd.DataFrame(infos).round(3))
""")

md("""
## 3. Deixa uma rede de fora: tipo 1 × tipo 2

Cada coluna é a rede que ficou de fora (o sistema nunca viu os erros dela).

| Método | O que é |
|---|---|
| Só P | a referência a superar |
| Regressão logística dentro | aprendida na própria rede (usa rótulos que, numa rede nova, não teríamos) |
| Regressão logística / boosting transferidos | aprendidos nas outras redes; referências não fuzzy |
| Fuzzy aditivo: tipo 1 dentro | o sistema fuzzy aprendido na própria rede (o melhor caso) |
| Fuzzy aditivo: tipo 1 transferido | regras e termos das outras redes |
| Fuzzy aditivo: tipo 1 + termos no alvo | regras das outras redes, termos ajustados com as features da rede nova (sem rótulos) |
| **Fuzzy aditivo: tipo 2 transferido** | termos = faixa entre as redes de origem |
| Fuzzy aditivo: tipo 2 + alvo no envelope | a faixa inclui também os termos da rede nova (sem rótulos) |

**O que conta como sucesso:** o tipo 2 transferido chega mais perto do "tipo 1 dentro" do que o tipo 1 transferido,
com o IC do ganho acima de zero; e a zona *incerto* concentra mais erros do que a zona *seguro*.
""")
code("""
res, zonas, regras, fous, preds = ft.deixa_uma_fora(tabs, lg.FEATURES, cfg.k_features, cfg.k_aditivo, cfg.forcar, cfg.q_alarme, cfg.n_boot)
princ = res[res.método.isin(ft.PRINCIPAIS)]
for met in ['AUROC', 'AUROC além de P', 'AURC (menor=melhor)']:
    print(met); display(princ.pivot_table(index='método', columns='rede alvo', values=met).reindex(ft.PRINCIPAIS).round(3))
display(princ[princ.método.str.contains(': tipo 2')][['rede alvo', 'método', 'ganho de AUROC sobre o tipo 1 transferido', 'IC 95% do ganho']])
""")

md("### As três zonas do tipo 2 (limiar escolhido nas redes de origem)")
code("display(zonas[zonas.sistema.str.startswith('Fuzzy aditivo')].round(3))")

md("""
## 4. As regras e as faixas

A tabela de faixas mostra, para cada termo, onde fica o pico em cada rede de origem: a distância entre eles é a incerteza do tipo 2.
""")
code("""
for (alvo, fam) in regras:
    if fam == 'Fuzzy aditivo':
        print(f'=== rede de fora: {alvo} ==='); display(fous[(alvo, fam)].round(3)); display(regras[(alvo, fam)])
""")

md("## 5. Figuras")
code("""
figs = ex.figuras(tabs, fous, preds, cfg.saida)
for f in figs: display(Image(filename=str(f)))
""")

md("## 6. Só para consulta: Wang-Mendel (fora do artigo)")
code("""
display(res[res.método.isin(ft.INTERNOS)].pivot_table(index='método', columns='rede alvo', values='AUROC').reindex(ft.INTERNOS).round(3))
display(zonas[zonas.sistema.str.startswith('Wang-Mendel')].round(3))
""")

md("## 7. Salvar (mande este arquivo)")
code("""
arq = ex.salvar(cfg, infos, hists, tabs, res, zonas, regras, fous, preds)
print('Salvo:', arq)
""")

nbf.write(nb, AQUI / "tipo2_entre_arquiteturas.ipynb")
print("ok")
