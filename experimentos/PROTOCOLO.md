# Protocolo experimental

Este arquivo registra **cada escolha** dos experimentos: o que foi feito, se é igual ao trabalho de referência
(Chen et al., AISTATS 2019), de onde veio a escolha e em que parte do código ela está. Serve para que outras pessoas
confiram e corrijam.

**Como corrigir:** edite a linha da tabela e, se mudar o código, cite o número do item (ex.: `C2`) na mensagem do commit.
Se uma correção mudar resultados, marque a coluna *Status* com `refazer` até rodar de novo.

**Legenda da coluna "Igual a Chen?":** `sim` · `não` (desvio, com o motivo) · `n/a` (Chen não fez isso) · `?` (não dá para saber pelo artigo).
**Legenda de Status:** `conferido` (verificado na fonte) · `a conferir` (escrito de memória ou suposição; precisa de verificação) · `refazer`.

Referência: Chen, T., Navrátil, J., Iyengar, V., Shanmugam, K. *Confidence Scoring Using Whitebox Meta-models with Linear Classifier Probes*. AISTATS 2019. [arXiv 1805.05396](https://arxiv.org/abs/1805.05396). O artigo **não publica** código, pesos da rede nem os índices do split.

Experimentos:
- **`classicos/`**: protocolo de Chen em CIFAR-10, CIFAR-100 e Tiny-ImageNet (notebook `classicos_comparacao.ipynb`).
- **`ham_enxuto/`**: o método no HAM10000 com a EfficientNet do BRACIS (resultados da rodada 5).
- **`ham_tipo2/`**: HAM10000 com três arquiteturas, fuzzy tipo 1 × tipo 2, deixando uma arquitetura de fora.

---

## A. Datasets

| # | Item | O que fizemos | Igual a Chen? | Por quê / fonte | Onde no código | Status |
|---|---|---|---|---|---|---|
| A1 | CIFAR-10 | 50.000 treino / 10.000 teste, 10 classes, 32×32, via `torchvision` | sim | Chen, seção 5 | `classicos.py: carregar` | conferido |
| A2 | CIFAR-100 | 50.000 / 10.000, 100 classes, 32×32, via `torchvision` | n/a | dataset clássico de detecção de falhas | `classicos.py: carregar` | conferido |
| A3 | Tiny-ImageNet | 100.000 treino (500 × 200 classes), 64×64. **TEST = conjunto `val` oficial** (10.000, com rótulos), porque o `test` oficial não tem rótulos | n/a | [cs231n](http://cs231n.stanford.edu/tiny-imagenet-200.zip); classes na ordem de `wnids.txt` | `classicos.py: carregar_tiny` | conferido |
| A4 | HAM10000 | 10.015 imagens dermatoscópicas; **maligno = mel, bcc, akiec**; benigno = o resto | n/a | Kaggle `kmader/skin-cancer-mnist-ham10000`; mesma definição do BRACIS 2025 | `ham_tipo2/redes.py: carregar_ham` | conferido |

## B. Divisão dos dados, ruído e fora da distribuição

| # | Item | O que fizemos | Igual a Chen? | Por quê / fonte | Onde no código | Status |
|---|---|---|---|---|---|---|
| B1 | Split nos datasets clássicos | Treino oficial embaralhado (`seed=0`) → TRAIN-BASE 60% (treina a rede) / TRAIN-META 20% (probes e meta-modelos) / DEV 20%. CIFAR: 30k/10k/10k; Tiny: 60k/20k/20k | tamanhos sim; **imagens não** | Chen dá só os tamanhos (30k/10k/10k/10k), não os índices | `classicos.py: dividir` | conferido |
| B2 | TEST nos clássicos | Teste oficial inteiro (CIFAR) / `val` oficial (Tiny). Sempre com rótulos limpos | sim (CIFAR-10) | Chen, seção 5 | notebook `classicos`, seção 2 | conferido |
| B3 | Ruído de rótulo | Cenário "ruidoso": 30% dos rótulos trocados por **outra** classe sorteada (`seed=1`), em TRAIN-BASE, TRAIN-META e DEV; TEST limpo | descrição sim; **sorteio não** | Chen, seção 5 | `classicos.py: ruido` | conferido |
| B4 | Fora da distribuição (OOD) | CIFAR-10 → teste do CIFAR-100 (como Chen); CIFAR-100 → teste do CIFAR-10; Tiny-ImageNet → sem OOD. Toda imagem OOD conta como "erro" | sim (CIFAR-10) | Chen, seção 5 | `classicos.py: DATASETS` | conferido |
| B5 | Split do HAM10000 (`ham_tipo2`) | Por **lesão** (a mesma lesão nunca em dois conjuntos), estratificado: treino 65% / val 10% / teste 25%, `seed=42`. Teste dividido em 2 metades, também por lesão | n/a | evitar vazamento entre fotos da mesma lesão | `ham_tipo2/redes.py: split_por_lesao` | conferido |
| B6 | Split do HAM10000 (`ham_enxuto`, rodada 5) | Split antigo do BRACIS: primeiras 3.000 imagens, 80/20, lesões repetidas tiradas da calibração. **Só 54 erros** na calibração | n/a | herdado do BRACIS; a ser substituído pelo B5 | `ham_enxuto/build.py`, seção 2 | refazer |

## C. Redes e treino

| # | Item | O que fizemos | Igual a Chen? | Por quê / fonte | Onde no código | Status |
|---|---|---|---|---|---|---|
| C1 | Rede dos clássicos | ResNet-32 **pré-ativação**: filtros 16/16/32/64, 5 unidades por estágio, ReLU vazado 0,1, atalho por *avg-pool* + canais completados com zero, padronização por imagem, BN + ReLU final (*unit_last*). ~0,46 M parâmetros | **?** | Chen usa "a ResNet do exemplo oficial do TensorFlow" sem dar detalhes; reconstruída a partir desse exemplo | `classicos.py: ResNetChen` | a conferir |
| C2 | Receita de treino dos clássicos | SGD momento 0,9, lote 128, decaimento de peso 2e-4, taxa 0,1 → 0,01 (40k passos) → 0,001 (60k) → 0,0001 (80k), total 90k passos; recorte aleatório com *padding* de 1/8 do lado (zeros) + espelho horizontal, por imagem; precisão mista (fp16) na GPU | **?** | hiperparâmetros do exemplo do TensorFlow, escritos de memória; Chen não os informa. Controle: acurácia no TEST do CIFAR-10 deve ficar perto de **90,4%** (limpo) e **77,4%** (ruidoso) | `classicos.py: treinar_chen` | a conferir |
| C3 | Tiny-ImageNet com a mesma ResNet-32 | Mesma rede e receita, entrada 64×64 sem adaptação | n/a | manter uma única arquitetura nos três datasets | `classicos.py: cenario` | conferido |
| C4 | Redes do HAM10000 (`ham_tipo2`) | EfficientNet-B0, ResNet-50, ViT-B/16 do `torchvision`, pesos ImageNet; cabeça do BRACIS (Dropout 0,4 → Linear 256 → ReLU → Dropout 0,3 → Linear 1); P = sigmoide | n/a | três famílias bem diferentes (CNN leve, CNN residual, transformer) | `ham_tipo2/redes.py: criar_rede` | conferido |
| C5 | Treino do HAM10000 | AdamW, OneCycle, taxa do backbone 3e-4 (EfficientNet) / 1e-4 (ResNet) / 3e-5 (ViT), cabeça 10×; 15 épocas, parada antecipada (4 épocas sem melhorar a AUROC da val); BCE com `pos_weight = √(benignos/malignos)`; aumento na GPU (giro qualquer, espelho, zoom 0,8–1, deslocamento, brilho, contraste); imagens redimensionadas para 256 e depois 224 | n/a | escolhas padrão de *fine-tuning* | `ham_tipo2/redes.py: treinar, LR_BACKBONE` | conferido |
| C6 | Limiar de decisão (HAM10000) | Escolhido na val, maximizando a acurácia balanceada (Youden) | n/a | classes desbalanceadas (~20% malignas) | `ham_tipo2/redes.py: escolher_limiar` | conferido |

## D. Leitura da rede (probes e features)

| # | Item | O que fizemos | Igual a Chen? | Por quê / fonte | Onde no código | Status |
|---|---|---|---|---|---|---|
| D1 | Pontos de leitura | Clássicos: convolução inicial + 15 unidades + *unit_last* = **17**. HAM: EfficientNet-B0 18 sub-blocos, ResNet-50 17 (maxpool + 16 bottlenecks), ViT-B/16 12 camadas | sim (17, CIFAR) | Chen cita 17 camadas lidas | `ResNetChen.pontos_de_leitura`; `ham_tipo2/redes.py: criar_rede` | conferido |
| D2 | Redução de cada ponto | Média espacial (CNN); token de classe (ViT) | **?** | Chen não detalha | `classicos.py: ler_ativacoes`; `ham_tipo2/leitura_geral.py: reduzir` | a conferir |
| D3 | Probes de Chen | Regressão logística (padronizada, C = 0,1) por ponto, treinada no TRAIN-META para prever o **rótulo** (ruidoso no cenário ruidoso) | ? | Chen: "linear classifier probes" (Alain & Bengio) treinadas no TRAIN-META | `classicos.py: Probes`, `cenario` | a conferir |
| D4 | Entrada dos meta-modelos de Chen | Probabilidades de cada probe em **ordem decrescente**; com 100/200 classes, só as **10 maiores** por probe | sim no CIFAR-10; **não** em 100/200 classes | ordenar tira a identidade da classe (como na análise de Chen); cortar em 10 evita 17 × 200 = 3.400 entradas | `classicos.py: ordenar, TOP_K` | conferido |
| D5 | Nossas probes | Mesma regressão logística, mas o alvo é a **decisão da rede** (não o rótulo) | n/a | a probe mede quando a rede "se decide" | `classicos.py: cenario` (`pr_nos`); `ham_tipo2/leitura_geral.py: fit` | conferido |
| D6 | Nossas features | Caminho (trocas de opinião, profundidade da decisão, entropia das probes, confiança no fim, salto final, integral de Choquet), estabilidade (vistas), familiaridade (k = 15 vizinhos no último ponto, distância cosseno), `prox_p`. No `ham_tipo2`, profundidade de 0 a 1 (`taxa_trocas`) | n/a | quatro motivos: resposta, caminho, estabilidade, familiaridade | `classicos.py: features_nossas`; `ham_tipo2/leitura_geral.py` | conferido |
| D7 | Vistas (estabilidade) | Clássicos: original, espelho horizontal, deslocamento de 2 px à direita, 2 px para baixo. HAM: original, espelho horizontal, espelho vertical, giro 180° | n/a | lesão de pele não tem orientação; objetos do CIFAR têm | `classicos.py: VISTAS`; `leitura_geral.py: VISTAS` | conferido |
| D8 | Imagens de referência (HAM) | Probes e banco de vizinhos treinados em 2.000 imagens do treino (na rodada 5, `ham_enxuto`, eram 1.000) | n/a | a rede já viu essas imagens | `ham_tipo2/experimento.py: ler_ou_carregar` | conferido |

## E. Meta-modelos e sistemas fuzzy

| # | Item | O que fizemos | Igual a Chen? | Por quê / fonte | Onde no código | Status |
|---|---|---|---|---|---|---|
| E1 | Softmax | Nota de erro = 1 − maior probabilidade | sim | Chen, Tabela 2 | `classicos.py: cenario` | conferido |
| E2 | Blackbox e Probes LR | `StandardScaler` + regressão logística (C = 1) | ? | Chen não dá hiperparâmetros | `classicos.py: lr` | a conferir |
| E3 | Blackbox e Probes GBM | `HistGradientBoosting` (300 iterações, taxa 0,05, profundidade 3, L2 = 1, parada antecipada) | **?** | Chen usa GBM sem detalhar a implementação | `classicos.py: gbm` | a conferir |
| E4 | Seleção de features (fuzzy) | Top-k pela AUROC no TRAIN-META, sem pares com correlação de Spearman ≥ 0,85; `prox_p` sempre entra no `ham_tipo2` | n/a | regras curtas e não redundantes | `regras_fuzzy.py: selecionar`; `fuzzy_t2.py: selecionar` | conferido |
| E5 | Fuzzy aditivo (**principal**) | 3 termos triangulares por feature (picos nos quantis 5/50/95%); regressão logística sobre as pertinências; k = 4 (clássicos) e k = 5 (`ham_tipo2`) | n/a | nas rodadas anteriores, empatou com a regressão logística e é legível | `regras_fuzzy.py: FuzzyAditivo`; `fuzzy_t2.py: FuzzyAditivo` | conferido |
| E6 | Árvore fuzzy | Fuzzy ID3, profundidade 2 | n/a | regras de 2 condições | `regras_fuzzy.py: ArvoreFuzzy` | conferido |
| E7 | Wang-Mendel | Roda, mas **fica fora do artigo e das tabelas principais** (abas `interno_*`) | n/a | decisão do autor | `fuzzy_t2.py: WangMendel` | conferido |
| E8 | Tipo 2 (`ham_tipo2`) | Termo = envelope (mín, máx) das curvas de cada rede de origem; nota pontual com as pertinências do meio da faixa; intervalo **exato** no fuzzy aditivo (nota linear nas pertinências) | n/a | a faixa representa a discordância entre redes | `fuzzy_t2.py: pert_t2, FuzzyAditivo.nota_t2` | conferido (testado contra força bruta) |
| E9 | Deixa uma arquitetura de fora | Aprende nas outras redes (metade 1 do teste) e avalia na rede de fora (metade 2), e troca; limiar das zonas escolhido nas redes de origem (20% acima) | n/a | testar se as regras se transferem | `fuzzy_t2.py: deixa_uma_fora` | conferido |

## F. Métricas

| # | Item | O que fizemos | Igual a Chen? | Por quê / fonte | Onde no código | Status |
|---|---|---|---|---|---|---|
| F1 | AUROC (erro × acerto) | Métrica principal | sim | Chen, Tabela 2 (AUC) | `classicos.py: metricas` | conferido |
| F2 | AUPR do erro, AURC | AURC ×1000 nos clássicos (como no FD-Shifts); menor é melhor | n/a | Jaeger et al., ICLR 2023 | `classicos.py: aurc`; `fuzzy_t2.py: aurc` | conferido |
| F3 | AUROC além de P | AUROC dentro de 5 faixas de P parecido, média ponderada | n/a | mede o que o método acrescenta ao softmax | `classicos.py: alem_de_p`; `fuzzy_t2.py: auroc_alem_de_p` | conferido |
| F4 | Intervalos de confiança | Bootstrap de 95%: 200 reamostras (clássicos), 1.000 pareadas (`ham_tipo2`) | n/a | — | `metricas`; `ic_diferenca` | conferido |
| F5 | Números publicados de Chen | CIFAR-10, Tabela 2: limpo Softmax 0,91, Blackbox LR/GBM 0,91, Probes LR 0,88, Probes GBM 0,91; ruidoso 0,74 / 0,79 / 0,80 / 0,87 / 0,88; OOD na mesma tabela | — | **comparação só aproximada**: rede, split e ruído não são os mesmos (B1, B3, C1, C2) | `classicos.py: CHEN` | a conferir (copiar da Tabela 2 de novo) |

## G. Reprodutibilidade

| # | Item | O que fizemos | Onde | Status |
|---|---|---|---|---|
| G1 | Sementes | split clássicos `seed=0`; ruído `seed=1`; HAM `seed=42`; `torch.manual_seed` fixo | notebooks e `Config` | conferido |
| G2 | Não determinismo | Treino em GPU com fp16 não é bit a bit reprodutível; repetir dá números um pouco diferentes | — | conferido |
| G3 | Versões | `requirements.txt` em cada pasta; torch ≥ 2.3 | `*/requirements.txt` | conferido |
| G4 | Retomada | Checkpoints, limiares e features ficam salvos e são reaproveitados se a sessão cair | `ham_tipo2/experimento.py`; `classicos` (`resnet32_*.pt`) | conferido |

## H. Resultados já obtidos (para saber o que precisa rodar de novo)

| # | Experimento | Situação |
|---|---|---|
| H1 | HAM10000, rodada 5 (`ham_enxuto`, split B6) | rodado: só P 0,799; regressão logística 0,832; fuzzy aditivo 0,820 (54 erros) |
| H2 | CIFAR-10, versão anterior (ResNet-32 pós-ativação, OneCycle, 16 pontos) | rodado: ruidoso, nossas features + LR 0,901 × softmax 0,800 × Probes GBM reimplementado 0,834; **antes** da correção da ordenação das entradas de Chen (D4) → refazer |
| H3 | Clássicos com C1/C2 (CIFAR-10, CIFAR-100, Tiny-ImageNet) | não rodado |
| H4 | `ham_tipo2` (três redes, tipo 1 × tipo 2) | não rodado |
