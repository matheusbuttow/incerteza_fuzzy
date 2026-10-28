"""Gera incerteza_fuzzy_enxuto.ipynb (módulos embutidos via %%writefile)."""
import nbformat as nbf
from pathlib import Path

AQUI = Path(__file__).parent
nb = nbf.v4.new_notebook(); C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Incerteza da rede lida por dentro + regras fuzzy (versão enxuta)

Todas as features vêm do **comportamento interno da rede** (nada medido nos pixels), em três eixos:
se ela muda de ideia ao longo das camadas, se muda de ideia entre versões espelhadas da mesma imagem,
e o que ela respondeu para casos parecidos do treino.
Sistemas fuzzy aprendem sozinhos regras do tipo *SE trocas_opiniao é ALTA ... ENTÃO ERRO*.

Saída: `incerteza_fuzzy_resultados.xlsx`. **Mande esse arquivo** a cada rodada: a aba `diagnostico_features` é a que eu uso para dizer quais features melhorar.
""")
for nome in ["leitura_rede.py", "regras_fuzzy.py"]:
    code(f"%%writefile {nome}\n" + (AQUI / nome).read_text())

md("## 1. Configuração")
code("""
METADATA_CSV = '/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_metadata.csv'
IMG_DIRS = ['/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_1',
            '/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_2']
MODEL_PT    = '/kaggle/input/models/matheusbuttow/modelo-bracis/pytorch/default/1/effnet_ham_best.pt'
SPLIT_CSV   = None   # do notebook de retreino (recomendado); None = split antigo do BRACIS
LIMIAR_JSON = None   # do notebook de retreino; None = usa LIMIAR
LIMIAR      = 0.5
N_REF       = 1000   # imagens do treino usadas como referência
K_FEATURES  = 4      # features por árvore
PROF        = 2      # profundidade da árvore fuzzy (2 = regras curtas e estáveis)
SUBBLOCOS   = True   # lê os 18 sub-blocos do EfficientNet (trajetória mais detalhada) em vez dos 9 estágios
SEED, BS    = 42, 32
""")

md("## 2. Dados e modelo")
code("""
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn, warnings
from pathlib import Path
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import leitura_rede as lr, regras_fuzzy as rf
warnings.filterwarnings('ignore'); pd.set_option('display.width', 220); pd.set_option('display.max_columns', 20)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(SEED)

meta = pd.read_csv(METADATA_CSV)
meta['is_malignant'] = meta.dx.isin({'mel', 'bcc', 'akiec'}).astype(int)
caminho = {a.stem: str(a) for d in IMG_DIRS for a in Path(d).glob('*.jpg')}
meta['image_path'] = meta.image_id.map(caminho); meta = meta.dropna(subset=['image_path'])
if LIMIAR_JSON: LIMIAR = json.load(open(LIMIAR_JSON))['limiar']
if SPLIT_CSV:
    meta = meta.merge(pd.read_csv(SPLIT_CSV)[['image_id', 'split']], on='image_id')
    treino, cal = meta[meta.split == 'train'], meta[meta.split == 'test']
else:
    m3 = meta.iloc[:3000]
    treino, cal = train_test_split(m3, test_size=0.2, stratify=m3.is_malignant, random_state=42)
    cal = cal[~cal.lesion_id.isin(treino.lesion_id)]          # tira lesões repetidas
ref = treino.sample(n=min(N_REF, len(treino)), random_state=SEED)
cal = cal.reset_index(drop=True)
print(f'referência: {len(ref)} | calibração: {len(cal)} imagens | limiar: {LIMIAR:.3f}')

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1), nn.Sigmoid())
model.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE)); model.to(DEVICE).eval()
tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
""")

md("## 3. Ler a rede por dentro")
code("""
leitor = lr.LeitorRede(model, limiar=LIMIAR, blocos=lr.blocos_efficientnet(model, SUBBLOCOS)).fit(lr.lotes_de_imagens(ref.image_path, tfm, DEVICE, BS), ref.is_malignant)
tab = lr.ler(leitor, cal, tfm, DEVICE, BS)
print(f'Erros da rede: {tab.erro.sum()} de {len(tab)} ({tab.erro.mean():.1%}) | a rede lê cada imagem em 4 versões (original e espelhadas)')
tab[['p', 'erro'] + lr.FEATURES].head()
""")

md("""
### Como a opinião da rede evolui ao longo das camadas

Para cada camada: chance de a rede, se parasse ali, já dar a resposta final. Linha = mediana; faixa = metade central das imagens.
Nos erros, espera-se uma curva mais baixa e irregular (a rede hesita e muda de ideia).
""")
code("""
import matplotlib.pyplot as plt
B = leitor.B; PB = tab[[f'probe_b{k}' for k in range(B)]].values
conf = np.where(tab.y_pred.values[:, None] == 1, PB, 1 - PB)
fig, ax = plt.subplots(figsize=(7, 3.6))
for nome, m, cor in [('acertos', tab.erro == 0, '#2a78d6'), ('erros', tab.erro == 1, '#eb6834')]:
    q = np.percentile(conf[m.values], [25, 50, 75], axis=0)
    ax.plot(range(1, B + 1), q[1], color=cor, lw=2, label=f'{nome} (n={m.sum()})')
    ax.fill_between(range(1, B + 1), q[0], q[2], color=cor, alpha=0.15)
ax.axhline(0.5, color='gray', ls='--', lw=1); ax.set_xlabel('bloco da rede'); ax.set_ylabel('concorda com a resposta final')
ax.set_ylim(0, 1); ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False); plt.tight_layout(); plt.show()
""")

md("""
### Exemplos com as fotos reais

Três lesões do seu conjunto: o acerto mais tranquilo, o erro com mais mudanças de ideia e o erro mais confiante (P longe do limiar).
Ao lado de cada foto, a curva de opinião da rede ao longo dos sub-blocos (abaixo de 0,5 = aquela camada pensava o contrário).
""")
code("""
from PIL import Image
ex = [('Acerto tranquilo', tab[tab.erro == 0].entropia_probes.idxmin()),
      ('Erro hesitante', tab[tab.erro == 1].trocas_opiniao.idxmax()),
      ('Erro confiante', tab[tab.erro == 1].prox_p.idxmin())]
fig, axes = plt.subplots(len(ex), 2, figsize=(9, 3.2 * len(ex)), gridspec_kw={'width_ratios': [1, 1.6]})
for (nome, i), (a1, a2) in zip(ex, axes):
    r = tab.loc[i]
    a1.imshow(Image.open(cal.image_path.iloc[i]).convert('RGB')); a1.axis('off')
    a1.set_title(f"{nome}\\n{r.image_id} | real: {'maligna' if r.y_true else 'benigna'} | rede: {'maligna' if r.y_pred else 'benigna'} (P={r.p:.2f})", fontsize=9, loc='left')
    a2.plot(range(1, B + 1), conf[i], marker='o', color='#b8325a' if r.erro else '#2a78d6')
    a2.axhline(0.5, color='gray', ls='--', lw=1); a2.set_ylim(0, 1); a2.set_xlabel('sub-bloco'); a2.set_ylabel('concorda com a resposta final')
    a2.set_title(f"mudou de ideia {int(r.trocas_opiniao)}x | decidiu a partir de {r.prof_decisao:.0%} da rede | firmeza no fim {r.conf_fim:.2f}", fontsize=9, loc='left')
    a2.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()
""")

md("""
## 4. Diagnóstico das features

Uma linha por feature. A coluna mais importante é **AUROC além de P**: o quanto a feature separa erros de acertos
entre imagens com P parecido, ou seja, o que ela traz de novo. Compare com a linha de `prox_p` (o próprio P).
""")
code("""
res, escolhas, oof = rf.comparar(tab, lr.FEATURES, k=K_FEATURES, prof=PROF, seed=SEED)
diag = rf.diagnostico(tab, lr.FEATURES, escolhas)
diag.round(3)
""")

md("""
## 5. Regras fuzzy × só P

Dois sistemas fuzzy, ambos com regras aprendidas sozinhas:
- **Árvore fuzzy**: regras com 2 condições (*SE A é ALTA E B é BAIXA ENTÃO ERRO*).
- **Fuzzy aditivo**: regras de 1 condição, cada uma com um peso, somadas (*SE salto_final é ALTA → +1,3 no risco*).
- **Wang-Mendel**: cada imagem propõe uma regra com 3 condições; as regras mais estáveis entre os três (com 3 features).
  Na rodada 4 ele chegou mais perto da regressão logística (0,83 contra 0,81 da árvore).
""")
code("""
display(res.round(3))
feats = rf.selecionar(tab, lr.FEATURES, K_FEATURES)
arv = rf.ArvoreFuzzy(PROF).fit(tab[feats].values, tab.erro.values, feats)
feats_wm = rf.selecionar(tab, lr.FEATURES, 3)                       # Wang-Mendel: 3 features
wm = rf.WangMendel().fit(tab[feats_wm].values, tab.erro.values, feats_wm)
adi = rf.FuzzyAditivo().fit(tab[feats].values, tab.erro.values, feats)
regras, regras_adi = arv.regras(), adi.regras()
est = rf.estabilidade(tab, feats, PROF)
print(f'Features: {feats} | estabilidade das regras da árvore: {est:.2f} (1 = sempre as mesmas)')
display(regras); display(regras_adi)
print(f'Wang-Mendel ({wm.n_regras} regras) com {feats_wm}:'); display(wm.regras())
""")

md("## 6. Salvar (mande este arquivo)")
code("""
resumo = pd.Series({'imagens de calibração': len(tab), 'erros da rede': int(tab.erro.sum()), 'limiar': LIMIAR,
                    'split': 'por lesão (retreino)' if SPLIT_CSV else 'antigo (BRACIS)', 'features da árvore': ', '.join(feats),
                    'estabilidade das regras': round(est, 3), 'blocos lidos': leitor.B, 'sub-blocos': SUBBLOCOS, 'imagens de referência': len(ref)})
with pd.ExcelWriter('incerteza_fuzzy_resultados.xlsx') as xw:
    resumo.to_frame('valor').to_excel(xw, sheet_name='resumo')
    diag.round(4).to_excel(xw, sheet_name='diagnostico_features', index=False)
    res.round(4).to_excel(xw, sheet_name='comparacao', index=False)
    regras.to_excel(xw, sheet_name='regras', index=False)
    regras_adi.to_excel(xw, sheet_name='regras_aditivo', index=False)
    wm.regras().to_excel(xw, sheet_name='regras_wang_mendel', index=False)
    tab.to_excel(xw, sheet_name='features_por_imagem', index=False)
print('Salvo: incerteza_fuzzy_resultados.xlsx')
""")

nbf.write(nb, AQUI / "incerteza_fuzzy_enxuto.ipynb")
print("ok")
