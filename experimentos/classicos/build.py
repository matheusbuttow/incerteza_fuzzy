"""Gera classicos_comparacao.ipynb (módulos embutidos via %%writefile)."""
import nbformat as nbf
from pathlib import Path

A = Path(__file__).parent; nb = nbf.v4.new_notebook(); C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip())); code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
md("""
# Nosso método × Chen et al. (AISTATS 2019) em três datasets clássicos

CIFAR-10, CIFAR-100 e Tiny-ImageNet, no protocolo de **"Confidence Scoring Using Whitebox Meta-models with Linear Classifier Probes"**.
Só o CIFAR-10 tem números publicados por Chen; nos outros dois, os métodos de Chen são reimplementados aqui e rodam na mesma rede.

O protocolo completo (o que é igual a Chen, o que difere e por quê) está em **`PROTOCOLO.md`** na raiz do repositório.

**Kaggle:** *Accelerator = GPU* e *Internet = On* (CIFAR pelo torchvision; Tiny-ImageNet baixado de cs231n.stanford.edu ou de um dataset anexado).
Tempo estimado numa T4: ~40–60 min por cenário no CIFAR e mais no Tiny-ImageNet (imagens 64×64); 3 datasets × 2 cenários.
Os modelos treinados ficam salvos (`resnet32_*.pt`): se a sessão cair, rode de novo e o treino é pulado.

Mande o arquivo `classicos_resultados.xlsx` no final.
""")
for f in ["regras_fuzzy.py", "classicos.py"]:
    code(f"%%writefile {f}\n" + (A / f).read_text())
md("## 1. Configuração")
code("""
DATASETS = ['cifar10', 'cifar100', 'tinyimagenet']
CENARIOS = [('limpo', 0.0), ('ruidoso', 0.3)]
MODELO   = 'chen'    # 'chen' = arquitetura e receita do exemplo do TensorFlow usado por Chen | 'nosso' = versão anterior
PASSOS   = 90_000    # receita de Chen: taxa 0,1 → 0,01 (40k) → 0,001 (60k) → 0,0001 (80k); se reduzir, as quedas acompanham
EPOCAS   = 60        # só para MODELO = 'nosso'
K_FUZZY  = 4         # features dos sistemas fuzzy (escolhidas no TRAIN-META)
COM_OOD  = True      # TEST + OOD: CIFAR-10 ↔ CIFAR-100 (Tiny-ImageNet: sem OOD)
DADOS    = './dados'
TINY_DIR = None      # pasta tiny-imagenet-200 de um dataset anexado; None = baixa (~240 MB)
""")
md("## 2. Rodar (para cada dataset: treina a ResNet, lê a rede, treina os meta-modelos no TRAIN-META, avalia no TEST)")
code("""
import numpy as np, pandas as pd, torch, warnings
import classicos as cc
warnings.filterwarnings('ignore'); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NENHUMA')
saidas, res, protocolo = cc.rodar_tudo(DATASETS, CENARIOS, MODELO, PASSOS, EPOCAS, K_FUZZY, COM_OOD, DADOS, TINY_DIR)
display(protocolo.round(3))
""")
md("""
## 3. Resultados (só o que vai para o artigo)

AUROC para detectar os erros da rede no TEST. Todas as linhas rodam na **mesma rede e nas mesmas imagens**, então a comparação entre linhas é par a par.
""")
code("""
ordem = [m.format(k=K_FUZZY) for m in cc.ORDEM]
art = res[res.uso == 'artigo']
for met in ['AUROC', 'AUROC além de P', 'AURC (x1000, menor melhor)']:
    print(met)
    display(art[art.conjunto == 'TEST'].pivot_table(index='método', columns=['dataset', 'cenário'], values=met, sort=False).reindex(ordem).round(3))
""")
md("### CIFAR-10: nossa reprodução × números publicados por Chen (Tabela 2)")
code("""
if (art.dataset == 'cifar10').any():
    c10 = art[art.dataset == 'cifar10'].pivot_table(index='método', columns=['cenário', 'conjunto'], values=['AUROC', 'AUROC Chen 2019 (publicado)'], sort=False)
    display(c10.reindex(ordem).round(3))
    print('Acurácia no TEST:', {f'{d}/{n}': round(v['acc'], 3) for (d, n), v in saidas.items() if d == 'cifar10'}, '| Chen: limpo 0.904, ruidoso 0.774')
""")
md("### Todas as métricas (com IC 95%)")
code("display(art.drop(columns=['AUROC Chen 2019 (publicado)', 'uso']).round(3))")
md("## 4. Regras do fuzzy aditivo e diagnóstico das features")
code("""
for (ds, nome), s in saidas.items():
    print(f'===== {ds} / {nome} ====='); display(s['regras_aditivo'])
for (ds, nome), s in saidas.items():
    if nome == 'ruidoso':
        print(f'diagnóstico das features: {ds} / ruidoso'); display(s['diag'].round(3))
""")
md("## 5. Só para consulta: Wang-Mendel (fora do artigo)")
code("""
wm = res[res.uso != 'artigo']
display(wm[wm.conjunto == 'TEST'].pivot_table(index='método', columns=['dataset', 'cenário'], values='AUROC').round(3))
""")
md("## 6. Salvar (mande este arquivo)")
code("print('Salvo:', cc.salvar(saidas, res, protocolo))")
nbf.write(nb, A / "classicos_comparacao.ipynb"); print("ok")
