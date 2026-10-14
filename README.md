# Incerteza da rede lida por dentro + regras fuzzy

Código completo do trabalho: treina as redes, lê cada rede por dentro, compara o nosso método (Wang-Mendel e fuzzy aditivo) com todos os métodos de outros artigos nas **mesmas imagens**, e gera planilha, gráficos e uma página HTML para analisar.

## 1. Instalar

Python 3.10 ou mais novo, com GPU NVIDIA e PyTorch com CUDA.

```bash
pip install -r requirements.txt
```

Se o `pip` instalar o PyTorch sem CUDA, instale a versão certa pelo site do PyTorch (pytorch.org → "Get Started") e depois rode o comando acima de novo.

## 2. Colocar o HAM10000

Baixe o HAM10000 (por exemplo, do Kaggle: `kmader/skin-cancer-mnist-ham10000`) e deixe assim:

```
incerteza_fuzzy/
  dados/
    ham10000/
      HAM10000_metadata.csv
      HAM10000_images_part_1/   (as .jpg)
      HAM10000_images_part_2/   (as .jpg)
```

O código procura as `.jpg` em qualquer subpasta de `dados/ham10000/`. CIFAR-10, CIFAR-100, SVHN, MNIST e o CIFAR-10H são **baixados sozinhos** na primeira execução (precisa de Internet).

## 3. Testar antes (5 a 10 minutos, sem GPU)

```bash
python rodar.py --teste-rapido
python relatorio.py --teste-rapido
```

Usa dados falsos e 1 época, só para conferir que tudo roda na sua máquina. Os números não significam nada. Saída em `resultados_teste_rapido/`.

## 4. Rodar de verdade

```bash
python rodar.py                 # tudo: 7 experimentos × 3 sementes
python relatorio.py             # junta tudo em resultados/relatorio.html e relatorio.xlsx
```

Para rodar por partes (recomendado — comece pelo HAM10000):

```bash
python rodar.py --exp ham10000
python rodar.py --exp cifar10_ruido30 cifar10
python rodar.py --exp cifar10_vgg16 cifar100_vgg16 svhn mnist
python relatorio.py
```

**É retomável:** se a máquina desligar, rode o mesmo comando de novo. O que já terminou (rede treinada, features, avaliação) é carregado do disco. O `relatorio.py` pode ser rodado a qualquer momento com o que já existir.

Outras opções: `--sementes 42 7` (escolher sementes) e `--refazer avaliacao` (refaz só a comparação, sem retreinar nem reler a rede — útil se mudar algo nos métodos).

### Tempo aproximado (por semente)

| Experimento | Rede | GPU tipo RTX 3090/4090 | GPU tipo T4 (Colab/Kaggle) |
|---|---|---|---|
| ham10000 (3000 lesões) | EfficientNet-B0, 15 épocas | ~20 min | ~40 min |
| cifar10, cifar10_ruido30 | ResNet-32, 100 épocas | ~35 min cada | ~1h30 cada |
| cifar10_vgg16, cifar100_vgg16 | VGG-16, 100 épocas | ~45 min cada | ~2h cada |
| svhn | CNN pequena, 30 épocas | ~15 min | ~30 min |
| mnist | MLP, 20 épocas | ~5 min | ~10 min |

Tudo (7 × 3 sementes): cerca de 10 h numa GPU boa. Para acelerar, diminua `EPOCAS` em `config.py` ou rode menos sementes primeiro.

## 5. O que sai

```
resultados/
  relatorio.html              ← abra no navegador: tabelas + todos os gráficos com explicação
  relatorio.xlsx              ← todas as tabelas
  figuras/*.png               ← gráficos em alta resolução para o artigo
  <experimento>/semente_<s>/  ← tudo de cada rodada (modelo, features por imagem, escores, regras, log)
```

Abas principais do `relatorio.xlsx`:

| Aba | O que tem |
|---|---|
| `visao_geral_AUROC` | AUROC (média ± desvio entre sementes) de cada método em cada experimento |
| `nossos_vs_so_P` | nossos dois métodos contra só P: ganho e em quantas sementes ele foi significativo |
| `por_experimento_media_desvio` | todas as métricas, média e desvio |
| `por_semente` | todas as métricas de cada rodada, com IC 95% do ganho |
| `numeros_publicados` | números publicados por Chen, ConfidNet e FD-Shifts ao lado dos nossos (ganho sobre o softmax de cada um) |
| `redes` | acurácia, erros, sensibilidade/especificidade (HAM) de cada rede treinada |
| `estabilidade_regras` | as regras se repetem quando os dados mudam? |
| `diagnostico_features` | o que cada feature traz de novo em relação a P |
| `regras_wang_mendel`, `regras_fuzzy_aditivo` | as regras aprendidas |
| `extra_dx_type`, `extra_multifotos`, `extra_cifar10h` | validações com incerteza real (veja abaixo) |

## 6. O que é comparado

Todos os métodos usam a **mesma rede congelada** e as **mesmas imagens de teste**; os que aprendem usam os **mesmos folds** de validação cruzada (5 folds no teste). Escore: maior = mais chance de a rede errar.

| Método | Referência | Aprende? |
|---|---|---|
| Só P (regressão logística com P e P²) | referência a superar | sim |
| Softmax (MSP), entropia do softmax | Hendrycks & Gimpel 2017 | não |
| Distância ao limiar (só HAM) | — | não |
| TTA (vistas) | — | não |
| MC Dropout (desvio e entropia) | Gal & Ghahramani 2016 | não |
| Mahalanobis | Lee et al. 2018 | não |
| Deep kNN | Papernot & McDaniel 2018 | não |
| Probes + GBM | Chen et al. 2019 | sim |
| Vizinhos em cada camada | Font et al. 2025 | sim |
| SVM na penúltima camada | Gorbett & Blanchard 2022 | sim |
| ConfidNet | Corbière et al. 2019 | sim |
| **NOSSO: Wang-Mendel** (P + 2 features internas) | proposto | sim |
| **NOSSO: Fuzzy aditivo** (P + 3 features internas) | proposto | sim |

Métricas: AUROC, AUPR-erro, AUPR-acerto, FPR@95%TPR, AURC, E-AURC, acurácia ao encaminhar 0–50% das imagens mais incertas, ganho de AUROC sobre só P com IC 95% (bootstrap pareado), média ± desvio entre sementes.

Experimentos (cada um reproduz o cenário de um artigo):

| Experimento | Cenário de | Rede |
|---|---|---|
| `ham10000` | seu trabalho (uma imagem por lesão, sem vazamento) | EfficientNet-B0 |
| `cifar10`, `cifar10_ruido30` | Chen et al. 2019 | ResNet-32 |
| `cifar10_vgg16`, `cifar100_vgg16` | ConfidNet 2019 / FD-Shifts 2023 | VGG-16 |
| `svhn` | ConfidNet 2019 / FD-Shifts 2023 | CNN pequena |
| `mnist` | ConfidNet 2019 | MLP |

## 7. Validações com incerteza real (nada sintético)

- **HAM10000 · `dx_type`:** os erros e a dúvida se concentram nas lesões **sem biópsia** (rótulo menos seguro)?
- **HAM10000 · várias fotos da mesma lesão:** as outras fotos das lesões de teste (nunca usadas no treino) passam pela rede. A dúvida medida na foto principal prevê se a rede muda de resposta em outra foto da mesma lesão?
- **CIFAR-10H:** cerca de 50 pessoas classificaram cada imagem de teste do CIFAR-10. Quando a rede tem dúvida, as pessoas também têm?

## 8. Arquivos do código

| Arquivo | O que faz |
|---|---|
| `config.py` | tudo que dá para mudar (caminhos, sementes, épocas, experimentos) |
| `dados.py` | carrega os conjuntos e as vistas |
| `redes.py` | as redes e o treino |
| `leitura.py` | lê a rede por dentro e calcula as features e os métodos clássicos |
| `fuzzy.py` | Wang-Mendel, fuzzy aditivo, seleção e diagnóstico das features |
| `comparacao.py` | todos os métodos, métricas e curvas |
| `extras.py` | as três validações com incerteza real |
| `rodar.py` | executa tudo (retomável) |
| `relatorio.py` | gera a planilha, os gráficos e o HTML |
