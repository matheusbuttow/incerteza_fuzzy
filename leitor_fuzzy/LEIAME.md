# leitor_fuzzy (extra, em teste)

Lê uma rede neural congelada qualquer (PyTorch) por dentro e diz **quando ela provavelmente alucinou e por quê**,
com as mesmas 11 características e o mesmo sistema fuzzy aditivo do artigo. Não precisa de código por arquitetura:
os pontos de leitura (blocos) são achados sozinhos.

```python
from leitor_fuzzy import LeitorFuzzy, treinar_rede

rede = treinar_rede(rede, (X_treino, y_treino), epocas=20)      # opcional: pode passar uma rede já treinada
lf = LeitorFuzzy(rede, vistas="imagem")                         # "imagem", "deslocamento", [funções] ou None
lf.fit((X_treino, y_treino), (X_val, y_val))                    # referência (treino) + calibração (dados não vistos)
risco = lf.risco((X_teste, y_teste))                            # chance de alucinação de cada exemplo
print(lf.diagnostico((X_teste, y_teste), nomes_classes))         # classes mais difíceis e o porquê
```

Os dados podem ser tuplas de tensores `(X, y)` ou um `DataLoader` que devolve `(x, y, ...)`.

## Tipo 1, tipo 2 e as três zonas

```python
lf = LeitorFuzzy(rede, vistas="imagem", tipo=2).fit((X_treino, y_treino), (X_val, y_val))
Z = lf.classificar((X_teste, y_teste))     # risco, risco_inf, risco_sup, zona, motivo (por exemplo)
print(lf.resumo_zonas(Z=Z))                # % dos exemplos, taxa de erro e % dos erros em cada zona
```

- **tipo=1**: uma nota por exemplo. **tipo=2**: um intervalo [risco_inf, risco_sup]; os termos BAIXA/MÉDIA/ALTA têm uma
  faixa de picos, tirada de `n_boot` reamostragens da calibração (com várias redes, veja `tipo1_vs_tipo2.py`).
- **Zonas**: SEGURO = a rede erra no máximo `meta_seguro` (padrão: 1/3 da taxa de erro na calibração);
  ALARME = erra pelo menos `meta_alarme` (50%); INCERTO = o resto. Os limiares são escolhidos na calibração
  (validação cruzada interna) com 95% de confiança (`garantia=True`). No tipo 2: seguro se o topo do intervalo
  está abaixo do limiar, alarme se a base está acima.
- `ajustar(F)`: refaz só a parte fuzzy a partir de `caracteristicas(calibracao)`, sem ler a rede de novo.

## O que cada parte faz

| arquivo | o que faz |
|---|---|
| `blocos.py` | acha os pontos de leitura: abre as pilhas de blocos repetidos (Sequential, camadas residuais, encoders) até `max_blocos` (20); ignora normalizações; a última camada é a cabeça |
| `leitor.py` | sondas por bloco (a sigmoide de cada bloco), as 11 características, seleção (p + 3), fuzzy aditivo, risco, explicação e diagnóstico por classe |
| `fuzzy.py` | partição triangular 5/50/95, fuzzy aditivo, seleção por AUROC além de p com corte de correlação 0,85 |
| `treino.py` | treino simples (Adam + entropia cruzada), opcional |
| `exemplo_mnist.py` | teste com 5.000 dígitos reais do MNIST (vêm dentro do pacote `mlxtend`) |
| `teste_tipos.py` | tipo 1 × tipo 2 com as três zonas, 3 redes × 5 sementes |

## Vistas (imagem transformada)

Padrão: **2 passadas** (a original + 1 transformada). A transformação não pode mudar a classe:
espelho horizontal para fotos (`"imagem"`), deslocamento de 1 pixel para dígitos e texto (`"deslocamento"`).
Para testar 3 ou mais, passe uma lista: `vistas=[f1, f2]`. Com `vistas=None` é uma passada só.

## Testes feitos

- **Detecção de blocos**: ResNet-32 e MLP do artigo = mesmos blocos escolhidos à mão; também funciona em
  ResNet-18/50, EfficientNet-B0 (18 blocos), ViT-B/16 (12 encoders), DenseNet-121 e MobileNet-V3.
- **Mesmas características do artigo**: lendo os mesmos blocos, as 11 características são idênticas às de `../leitura.py`.
- **MNIST real (5.000 dígitos)**: o fuzzy supera a confiança do softmax na CNN e na MLP; as confusões apontadas
  (8→5, 3→5, 4→9, 7→9) são as clássicas do MNIST.
