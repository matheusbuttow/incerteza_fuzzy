"""
redes.py — as redes de cada artigo, todas com a mesma "interface" para podermos lê-las por dentro:

  rede.blocos   lista dos blocos cuja saída é lida (um probe por bloco)
  rede.corpo(x) vetor da penúltima camada (o que entra na cabeça)
  rede.cabeca   cabeça de classificação COM Dropout (permite MC Dropout sem retreinar)
  rede(x)       logits: (n, C) em multiclasse; (n, 1) no binário (P = sigmoide)

Arquiteturas: EfficientNet-B0 (HAM10000, mesma cabeça do BRACIS), ResNet-32 (Chen 2019),
VGG-16 com BatchNorm (ConfidNet / FD-Shifts), CNN pequena (SVHN) e MLP (MNIST).
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

import config as C


def probabilidades(logits):
    """(n, C) probabilidades. No binário, a coluna 1 é P(maligna)."""
    if logits.shape[1] == 1:
        p1 = torch.sigmoid(logits[:, 0])
        return torch.stack([1 - p1, p1], 1)
    return torch.softmax(logits, 1)


# ----------------------------------------------------------------------------- arquiteturas
class Rede(nn.Module):
    def forward(self, x):
        return self.cabeca(self.corpo(x))


class EffNetB0(Rede):
    def __init__(self, pretreinada=True, n_classes=2):
        super().__init__()
        from torchvision import models
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretreinada else None)
        self.features, self.pool = m.features, m.avgpool
        saida = 1 if n_classes == 2 else n_classes                    # binário: uma saída (sigmoide); senão, softmax
        self.cabeca = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, saida))
        sub = [self.features[0]]
        for est in list(self.features.children())[1:-1]:
            sub += list(est.children())
        self.blocos = sub + [self.features[-1]]                       # 18 sub-blocos

    def corpo(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


class BlocoBasico(nn.Module):
    def __init__(self, ent, sai, passo):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(ent, sai, 3, passo, 1, bias=False), nn.BatchNorm2d(sai)
        self.c2, self.b2 = nn.Conv2d(sai, sai, 3, 1, 1, bias=False), nn.BatchNorm2d(sai)
        self.atalho = nn.Sequential() if passo == 1 and ent == sai else nn.Sequential(nn.Conv2d(ent, sai, 1, passo, bias=False), nn.BatchNorm2d(sai))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.atalho(x))


class ResNet32(Rede):
    def __init__(self, n_classes):
        super().__init__()
        self.inicio = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        camadas, ent = [], 16
        for sai, passo in [(16, 1), (32, 2), (64, 2)]:
            for i in range(5):
                camadas.append(BlocoBasico(ent, sai, passo if i == 0 else 1)); ent = sai
        self.camadas = nn.Sequential(*camadas)
        self.cabeca = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, n_classes))
        self.blocos = [self.inicio] + list(self.camadas)               # 16 blocos

    def corpo(self, x):
        return F.adaptive_avg_pool2d(self.camadas(self.inicio(x)), 1).flatten(1)


class VGG16(Rede):
    CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

    def __init__(self, n_classes):
        super().__init__()
        blocos, ent = [], 3
        for i, v in enumerate(self.CFG):
            if v == "M":
                continue
            b = [nn.Conv2d(ent, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU()]
            if i + 1 < len(self.CFG) and self.CFG[i + 1] == "M":
                b.append(nn.MaxPool2d(2))
            blocos.append(nn.Sequential(*b)); ent = v
        self.conv = nn.Sequential(*blocos)
        self.cabeca = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, n_classes))
        self.blocos = list(self.conv)                                  # 13 blocos

    def corpo(self, x):
        return F.adaptive_avg_pool2d(self.conv(x), 1).flatten(1)        # 32×32 → 1×1 (igual a antes); 64×64 → média de 2×2


class CNNPequena(Rede):
    def __init__(self, n_classes, canais=3):
        super().__init__()
        def b(e, s, pool):
            return nn.Sequential(nn.Conv2d(e, s, 3, padding=1), nn.BatchNorm2d(s), nn.ReLU(), *( [nn.MaxPool2d(2)] if pool else []))
        self.conv = nn.Sequential(b(canais, 32, False), b(32, 32, True), b(32, 64, False), b(64, 64, True), b(64, 128, True))
        self.cabeca = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))
        self.blocos = list(self.conv)

    def corpo(self, x):
        return F.adaptive_avg_pool2d(self.conv(x), 1).flatten(1)


class MLP(Rede):
    def __init__(self, n_classes, entrada=784):
        super().__init__()
        self.h1 = nn.Sequential(nn.Flatten(), nn.Linear(entrada, 512), nn.ReLU())
        self.h2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.cabeca = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, n_classes))
        self.blocos = [self.h1, self.h2]

    def corpo(self, x):
        return self.h2(self.h1(x))


class MobileNetV2P(Rede):
    """MobileNetV2 treinada do zero para imagens pequenas (32×32 e 64×64): o 1º passo não reduz a imagem."""
    def __init__(self, n_classes):
        super().__init__()
        from torchvision import models
        m = models.mobilenet_v2(weights=None, num_classes=n_classes)
        m.features[0][0].stride = (1, 1)
        self.features = m.features
        self.cabeca = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, n_classes))
        self.blocos = list(self.features)                              # 19 blocos (entrada + 17 invertidos + saída)

    def corpo(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


class ResNet50(Rede):
    """ResNet-50 pré-treinada no ImageNet (imagens 224×224)."""
    def __init__(self, n_classes, pretreinada=True):
        super().__init__()
        from torchvision import models
        m = models.resnet50(weights="IMAGENET1K_V2" if pretreinada else None)
        self.inicio = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.camadas = nn.Sequential(m.layer1, m.layer2, m.layer3, m.layer4)
        saida = 1 if n_classes == 2 else n_classes
        self.cabeca = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, saida))
        self.blocos = [self.inicio] + [b for c in self.camadas for b in c]   # 17 blocos

    def corpo(self, x):
        return F.adaptive_avg_pool2d(self.camadas(self.inicio(x)), 1).flatten(1)


class ViTB16(Rede):
    """ViT-B/16 pré-treinado no ImageNet (imagens 224×224). Cada camada do encoder é um bloco; a leitura faz a média dos tokens."""
    def __init__(self, n_classes, pretreinada=True):
        super().__init__()
        from torchvision import models
        self.vit = models.vit_b_16(weights="IMAGENET1K_V1" if pretreinada else None)
        self.vit.heads = nn.Identity()
        saida = 1 if n_classes == 2 else n_classes
        self.cabeca = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, saida))
        self.blocos = list(self.vit.encoder.layers)                    # 12 blocos

    def corpo(self, x):
        return self.vit(x)                                             # token de classe, após a normalização final


def criar(arq, n_classes):
    pre = not C.TESTE_RAPIDO
    return {"effnet_b0": lambda: EffNetB0(pretreinada=pre, n_classes=n_classes),
            "resnet50": lambda: ResNet50(n_classes, pretreinada=pre), "vit_b_16": lambda: ViTB16(n_classes, pretreinada=pre),
            "resnet32": lambda: ResNet32(n_classes), "vgg16": lambda: VGG16(n_classes), "mobilenet_v2": lambda: MobileNetV2P(n_classes),
            "cnn": lambda: CNNPequena(n_classes), "mlp": lambda: MLP(n_classes)}[arq]()


# ----------------------------------------------------------------------------- treino e previsão
@torch.no_grad()
def prever(rede, carregador, vista=None, disp="cpu"):
    rede.eval(); saida = []
    for x, _, _ in carregador:
        x = x.to(disp, non_blocking=True)
        saida.append(probabilidades(rede(vista(x) if vista else x).float()).cpu())
    return torch.cat(saida).numpy()


def treinar(rede, conj, arq, semente, disp, log=print):
    """Treina a rede só no treino. No binário (HAM), escolhe a melhor época e o limiar (Youden) na validação."""
    from dados import carregador
    torch.manual_seed(semente); np.random.seed(semente)
    epocas = 1 if C.TESTE_RAPIDO else (getattr(conj, "epocas", None) or C.EPOCAS[arq])
    lote = C.LOTE[arq]
    dl = carregador(conj.treino, lote, embaralhar=True)
    rede.to(disp)
    if arq in ("effnet_b0", "resnet50", "vit_b_16"):                   # pré-treinadas: ajuste fino com AdamW
        opt = torch.optim.AdamW(rede.parameters(), lr={"effnet_b0": 3e-4, "resnet50": 1e-4, "vit_b_16": 3e-5}[arq], weight_decay=1e-4)
    elif arq in ("resnet32", "vgg16", "mobilenet_v2"):
        opt = torch.optim.SGD(rede.parameters(), lr={"resnet32": 0.1, "vgg16": 0.05, "mobilenet_v2": 0.05}[arq],
                              momentum=0.9, weight_decay=5e-4 if arq != "mobilenet_v2" else 4e-5, nesterov=True)
    else:
        opt = torch.optim.Adam(rede.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=epocas * len(dl), pct_start=0.15)
    amp = disp == "cuda"
    escala = torch.amp.GradScaler("cuda", enabled=amp)
    if conj.binario:
        y_tr = np.asarray([conj.treino.y[i] for i in range(len(conj.treino))]) if hasattr(conj.treino, "y") else None
        peso_pos = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)) if y_tr is not None else 1.0
    melhor, estado = -np.inf, None
    for ep in range(epocas):
        rede.train(); soma, n = 0.0, 0
        for x, y, _ in dl:
            x, y = x.to(disp, non_blocking=True), y.to(disp, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                z = rede(x)
                if conj.binario:
                    yf = y.float()
                    loss = F.binary_cross_entropy_with_logits(z[:, 0].float(), yf, weight=torch.where(yf == 1, peso_pos, 1.0))
                else:
                    loss = F.cross_entropy(z.float(), y)
            opt.zero_grad(set_to_none=True)
            escala.scale(loss).backward(); escala.step(opt); escala.update(); sched.step()
            soma += loss.item() * len(y); n += len(y)
        msg = f"  época {ep + 1:3d}/{epocas} | perda {soma / n:.4f}"
        if conj.val is not None:
            pv = prever(rede, carregador(conj.val, lote), disp=disp)
            auc = roc_auc_score(conj.y_val, pv[:, 1]) if conj.binario else float((pv.argmax(1) == conj.y_val).mean())
            msg += f" | {'AUROC' if conj.binario else 'acurácia'} validação {auc:.3f}"
            if auc > melhor:
                melhor, estado = auc, copy.deepcopy(rede.state_dict())
        log(msg)
    if estado is not None:
        rede.load_state_dict(estado)
    limiar = None
    if conj.binario:
        pv = prever(rede, carregador(conj.val, lote), disp=disp)[:, 1]
        fpr, tpr, lim = roc_curve(conj.y_val, pv)
        limiar = float(np.clip(lim[np.argmax(tpr - fpr)], 1e-3, 1 - 1e-3))
    return {"limiar": limiar}
