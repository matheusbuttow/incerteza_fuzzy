"""
config.py — tudo que você pode querer mudar está aqui.

Cada EXPERIMENTO reproduz o cenário de um artigo, para aplicarmos os métodos deles nos nossos dados
e o nosso método nos cenários deles:
  ham10000          o seu trabalho (EfficientNet-B0, benigno × maligno, uma imagem por lesão, 3000 lesões)
  ham10000_todas    o mesmo com TODAS as ~7.400 lesões (teste ~1.480 imagens) e 5 sementes
  cifar10           Chen et al. 2019 (ResNet-32, CIFAR-10 limpo)
  cifar10_ruido30   Chen et al. 2019 (ResNet-32, 30% dos rótulos de treino trocados)
  cifar10_vgg16     ConfidNet 2019 / FD-Shifts 2023 (VGG-16, CIFAR-10)
  cifar100_vgg16    ConfidNet 2019 / FD-Shifts 2023 (VGG-16, CIFAR-100)
  svhn              ConfidNet 2019 / FD-Shifts 2023 (CNN pequena, SVHN)
  mnist             ConfidNet 2019 (MLP, MNIST)
"""
from pathlib import Path

# ----------------------------------------------------------------------------- caminhos
DADOS = Path("dados")                                  # CIFAR, SVHN e MNIST são baixados aqui automaticamente
# Pasta que contém o HAM10000_metadata.csv. Usa a primeira que existir; para outro lugar, troque por Path("...").
HAM_DIR = next((p for p in [Path("dados/ham10000"), Path("dados/archive")] if (p / "HAM10000_metadata.csv").exists()),
               Path("dados/ham10000"))
ANIMAL_DIR = Path("dados/animal10n")    # ANIMAL-10N: pastas training/ e testing/ com arquivos "<classe>_<n>.jpg"
CHAOYANG_DIR = Path("dados/chaoyang")   # Chaoyang: train.json e test.json (lista de {"name": caminho, "label": classe}) + imagens
SAIDA = Path("resultados")

# ----------------------------------------------------------------------------- o que rodar
SEMENTES = [42, 7, 2026]                               # cada semente: novo sorteio dos dados + novo treino da rede
# Seleção do artigo: um cenário por domínio. CIFAR-10 limpo e CIFAR-10N usam AS MESMAS imagens (só os rótulos de treino
# mudam), isolando o efeito do ruído real; ANIMAL-10N (web) e Chaoyang (médico) trazem domínios diferentes.
# Fora do padrão (definidos abaixo, ainda rodáveis com --exp; relatório com --todos): cifar10_vgg16, cifar100_vgg16, mnist,
# cifar10n_aggre/random1, cifar100n, ham10000*, cifar10_ruido* (ruído sintético), svhn.
EXPERIMENTOS_PADRAO = ["cifar10", "cifar10n_worst", "animal10n", "chaoyang"]

EXPERIMENTOS = {
    "ham10000":        dict(dados="ham10000", arq="effnet_b0", ruido=0.0, artigo="HAM10000 (mesma configuração do seu BRACIS 2025)"),
    "ham10000_todas":  dict(dados="ham10000", arq="effnet_b0", ruido=0.0, artigo="HAM10000 (mesma configuração do seu BRACIS 2025)",
                            n_lesoes=None, sementes=[42, 7, 2026, 1, 123]),   # todas as lesões, 5 sementes
    "cifar10":         dict(dados="cifar10",  arq="resnet32",  ruido=0.0, artigo="Chen et al. 2019"),
    "cifar10_ruido30": dict(dados="cifar10",  arq="resnet32",  ruido=0.3, artigo="Chen et al. 2019"),
    # curva de ruído: o ganho cresce com a fração de rótulos de treino trocados? (rodar_ruido.sh; fora do padrão)
    "cifar10_ruido10": dict(dados="cifar10",  arq="resnet32",  ruido=0.1, artigo="curva de ruído (ResNet-32)"),
    "cifar10_ruido20": dict(dados="cifar10",  arq="resnet32",  ruido=0.2, artigo="curva de ruído (ResNet-32)"),
    "cifar10_ruido40": dict(dados="cifar10",  arq="resnet32",  ruido=0.4, artigo="curva de ruído (ResNet-32)"),
    # ruído REAL: rótulos de treino dados por pessoas (CIFAR-10N / CIFAR-100N, Wei et al., ICLR 2022); teste com os rótulos oficiais
    "cifar10n_aggre":   dict(dados="cifar10",  arq="resnet32", ruido="humano:aggre_label",   artigo="CIFAR-10N (voto de 3 pessoas, ~9%)"),
    "cifar10n_random1": dict(dados="cifar10",  arq="resnet32", ruido="humano:random_label1", artigo="CIFAR-10N (1 pessoa, ~17%)"),
    "cifar10n_worst":   dict(dados="cifar10",  arq="resnet32", ruido="humano:worse_label",   artigo="CIFAR-10N (pior de 3, ~40%)"),
    "cifar100n":        dict(dados="cifar100", arq="vgg16",    ruido="humano:noisy_label",   artigo="CIFAR-100N (1 pessoa, ~40%)"),
    # ruído REAL da internet: 5 pares de animais parecidos, ~8% dos rótulos de treino errados; teste verificado (Song et al., ICML 2019)
    "animal10n":        dict(dados="animal10n", arq="vgg16",   ruido=0.0, epocas=60, artigo="ANIMAL-10N (ruído da web, ~8%)"),
    # ruído REAL médico: histopatologia colorretal, rótulos de treino com discordância entre patologistas; teste por consenso
    "chaoyang":         dict(dados="chaoyang", arq="effnet_b0", ruido=0.0, artigo="Chaoyang (patologistas discordam)"),
    "cifar10_vgg16":   dict(dados="cifar10",  arq="vgg16",     ruido=0.0, artigo="ConfidNet 2019 / FD-Shifts 2023"),
    "cifar100_vgg16":  dict(dados="cifar100", arq="vgg16",     ruido=0.0, artigo="ConfidNet 2019 / FD-Shifts 2023"),
    "svhn":            dict(dados="svhn",     arq="cnn",       ruido=0.0, artigo="ConfidNet 2019 / FD-Shifts 2023"),
    "mnist":           dict(dados="mnist",    arq="mlp",       ruido=0.0, artigo="ConfidNet 2019"),
}

# ----------------------------------------------------------------------------- HAM10000
HAM_N_LESOES = 3000            # lesões diferentes (uma imagem por lesão); None = todas (~7.400)
HAM_FRAC_VAL, HAM_FRAC_TESTE = 0.10, 0.20
HAM_TAM = 224                  # tamanho da imagem (pixels)

# ----------------------------------------------------------------------------- treino das redes
EPOCAS = {"effnet_b0": 15, "resnet32": 100, "vgg16": 100, "cnn": 30, "mlp": 20}
LOTE = {"effnet_b0": 32, "resnet32": 128, "vgg16": 128, "cnn": 128, "mlp": 128}
TRABALHADORES = 4              # processos do DataLoader (0 se der erro no Windows)

# ----------------------------------------------------------------------------- leitura da rede e métodos
N_REF = 5000                   # imagens de treino usadas como referência (probes, vizinhos, Mahalanobis)
PCA_PROBES = 256               # reduz camadas largas antes dos probes (acelera; None = sem PCA)
K_VIZINHOS = 15                # Deep kNN / nossas features de vizinhos
K_FONT = 10                    # vizinhos por camada (Font et al. 2025)
N_REF_FONT = 2000              # referência usada por Font et al. (Bray-Curtis em todas as camadas é lento)
T_MC = 30                      # repetições do MC Dropout
N_FOLDS = 5                    # validação cruzada dos métodos que aprendem (no conjunto de teste)
N_BOOT = 1000                  # reamostragens para os intervalos de confiança
FEATURES_WM = 3                # Wang-Mendel: P + 2 features
FEATURES_ADITIVO = 4           # fuzzy aditivo: P + 3 features
ENCAMINHAR = [round(0.05 * i, 2) for i in range(11)]   # 0%, 5%, ..., 50% encaminhados ao especialista
# três zonas (classificar cada resposta da rede em SEGURO / INCERTO / ALARME), limiares escolhidos no treino de cada dobra
ZONA_META_SEGURO_REL = 1 / 3   # SEGURO: a rede erra no máximo 1/3 da sua taxa de erro normal (ex.: 15% → 5%)
ZONA_META_ALARME = 0.50        # ALARME: a rede erra em pelo menos 50% dessas imagens
ZONA_GARANTIA = True           # True: vale com 95% de confiança (limite de Clopper–Pearson); False: taxa observada

# ----------------------------------------------------------------------------- teste rápido (sem GPU, dados falsos)
TESTE_RAPIDO = False           # rodar.py --teste-rapido liga isto
