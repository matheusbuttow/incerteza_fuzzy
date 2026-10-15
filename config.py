"""
config.py — tudo que você pode querer mudar está aqui.

Cada EXPERIMENTO reproduz o cenário de um artigo, para aplicarmos os métodos deles nos nossos dados
e o nosso método nos cenários deles:
  ham10000          o seu trabalho (EfficientNet-B0, benigno × maligno, uma imagem por lesão)
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
SAIDA = Path("resultados")

# ----------------------------------------------------------------------------- o que rodar
SEMENTES = [42, 7, 2026]                               # cada semente: novo sorteio dos dados + novo treino da rede
EXPERIMENTOS_PADRAO = ["ham10000", "cifar10", "cifar10_ruido30", "cifar10_vgg16", "cifar100_vgg16", "svhn", "mnist"]

EXPERIMENTOS = {
    "ham10000":        dict(dados="ham10000", arq="effnet_b0", ruido=0.0, artigo="Seu trabalho (BRACIS 2025)"),
    "cifar10":         dict(dados="cifar10",  arq="resnet32",  ruido=0.0, artigo="Chen et al. 2019"),
    "cifar10_ruido30": dict(dados="cifar10",  arq="resnet32",  ruido=0.3, artigo="Chen et al. 2019"),
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

# ----------------------------------------------------------------------------- teste rápido (sem GPU, dados falsos)
TESTE_RAPIDO = False           # rodar.py --teste-rapido liga isto
