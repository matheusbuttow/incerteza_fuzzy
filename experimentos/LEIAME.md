# Experimentos: protocolo de Chen, três datasets clássicos e fuzzy tipo 2

Um sistema fuzzy observa **como** uma rede já treinada chegou à resposta (probes em cada bloco, estabilidade,
familiaridade) e avisa quando ela provavelmente errou. Aplicação principal: câncer de pele (HAM10000).

**Antes de tudo, leia [`PROTOCOLO.md`](PROTOCOLO.md):** a tabela com cada escolha dos experimentos, o que é igual
ou diferente do trabalho de referência (Chen et al., 2019) e o que ainda precisa ser conferido.

| pasta | experimento | Jupyter | Python (terminal) |
|---|---|---|---|
| `classicos/` | protocolo de Chen em CIFAR-10, CIFAR-100 e Tiny-ImageNet | `classicos_comparacao.ipynb` | `python rodar.py` |
| `ham_enxuto/` | o método no HAM10000 com a EfficientNet do BRACIS (rodada 5) | `incerteza_fuzzy_enxuto.ipynb` | `python rodar.py --metadata ... --imagens ... --modelo effnet_ham_best.pt` |
| `ham_tipo2/` | três arquiteturas no HAM10000; fuzzy tipo 1 × tipo 2, deixando uma rede de fora | `tipo2_entre_arquiteturas.ipynb` | `python rodar.py --metadata ... --imagens ...` (ver `ham_tipo2/LEIAME.md`) |

O notebook e o script de cada pasta usam os mesmos módulos `.py`, então dão o mesmo resultado
(o script só não desenha os gráficos do `ham_enxuto`). Rode os scripts de dentro da pasta (`cd classicos`).

Em cada pasta, `build.py` gera o notebook a partir dos módulos `.py`; edite os módulos e rode `python build.py`.
O Wang-Mendel roda só para consulta e fica fora das tabelas principais (abas `interno_*`).
