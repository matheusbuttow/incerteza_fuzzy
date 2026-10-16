# Teste 1 — resultados congelados (16/10/2026)

Código: commit `f56b043` (tag `teste1`). Máquina: RTX 4070 Ti. Planilha completa: `relatorio.xlsx` (nesta pasta).

8 experimentos: HAM10000 com 3000 lesões (3 sementes) e com todas as lesões (5 sementes: 42, 7, 2026, 1, 123);
CIFAR-10 limpo e com 30% de ruído (ResNet-32), CIFAR-10 e CIFAR-100 (VGG-16), SVHN (CNN pequena), MNIST (MLP) — 3 sementes cada (42, 7, 2026).
Todos os métodos usam a mesma rede congelada e as mesmas imagens de teste; os que aprendem usam os mesmos 5 folds.

## AUROC para detectar os erros da rede (média ± desvio entre sementes)

| Cenário | Melhor dos outros | Fuzzy aditivo | Aditivo sem vistas (1 passagem) | Softmax | MC Dropout (entropia) |
|---|---|---|---|---|---|
| HAM10000 completo (5 sementes) | Chen 0,909 | **0,916 ± 0,012** | 0,914 ± 0,016 | 0,865 | 0,864 |
| HAM10000 3000 lesões | Chen 0,896 | **0,904 ± 0,044** | 0,901 | 0,819 | 0,818 |
| CIFAR-10 ruído 30% | Gorbett 0,880 | **0,903 ± 0,003** | 0,896 | 0,797 | 0,779 |
| CIFAR-100 VGG-16 | Chen 0,871 | **0,886 ± 0,002** | 0,875 | 0,866 | 0,864 |
| CIFAR-10 ResNet-32 | TTA 0,931 | **0,936 ± 0,002** | 0,919 | 0,928 | 0,926 |
| CIFAR-10 VGG-16 | TTA 0,931 | **0,934 ± 0,002** | 0,922 | 0,915 | 0,910 |
| SVHN | softmax 0,926 | 0,926 (empate) | 0,925 | 0,926 | 0,918 |
| MNIST | TTA 0,986 | 0,982 (empate prático) | 0,981 | 0,985 | 0,984 |

## Conclusões

1. Fuzzy aditivo: melhor ou empatado com o melhor nos 8 cenários; menor E-AURC em quase todos.
2. Contra Chen et al. 2019: vence com significância (IC 95%) em 18 de 26 rodadas (cenário × semente) e não perde em nenhuma.
3. Contra os métodos clássicos de incerteza (softmax, MC Dropout, TTA): +0,05 no HAM completo, +0,10 no CIFAR com ruído.
4. Sem as vistas (uma passagem pela rede) o resultado se mantém onde importa: HAM (0,914) e CIFAR com ruído (0,896).
   No CIFAR limpo as vistas ajudam (sem elas: 0,919 < softmax 0,928).
5. No HAM, contra "Só P" e Chen o aditivo empata na prática (encaminhando 20%: 106, 107 e 105 de 139 erros por 1.000 lesões);
   a vantagem do aditivo ali é a explicação por regras, não o desempenho.
6. Wang-Mendel é instável: perde para Chen com significância no CIFAR-100 (3/3), HAM completo (4/5) e SVHN (2/3) — sai como método principal.
7. HAM: a rede erra 26% nas lesões com biópsia (histo) e 2,5% nas sem biópsia — a dúvida da rede coincide com a dúvida clínica.
8. Regras do aditivo estáveis: correlação dos pesos entre reamostragens 0,91–0,94 (HAM completo), 0,90–0,99 (CIFAR).

## HAM10000 completo: em cada 1.000 lesões (rede erra ~139), quantos erros o médico encontra revisando 200

| Quem escolhe | Erros encontrados | Erros que escapam |
|---|---|---|
| Ao acaso | 28 | 111 |
| MC Dropout / softmax | 82 | 57 |
| Chen et al. 2019 | 105 | 34 |
| Só P | 107 | 32 |
| Fuzzy aditivo | 106 | 33 |

## Pendências conhecidas

- No HAM, a vitória sobre Chen é significativa em só 1–2 de 5 sementes.
- Mahalanobis sai invertido no HAM (AUROC ~0,18): os erros são lesões típicas (falsos positivos de benignas comuns).
- Na validação "várias fotos da mesma lesão", quem melhor prevê a mudança de resposta é a distância ao limiar (~0,72); o aditivo fica em ~0,64.
