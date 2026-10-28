"""
redes.py — dados do HAM10000, split por lesão, várias arquiteturas com a MESMA cabeça, treino e limiar.

Tudo o que muda de uma arquitetura para outra fica aqui:
  - como montar a rede (backbone pré-treinado no ImageNet + cabeça igual à do BRACIS);
  - quais são os PONTOS DE LEITURA (a saída de cada bloco), onde as probes vão ser colocadas.
O resto do código (leitura, fuzzy, avaliação) não sabe qual rede está olhando.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import models

MALIGNAS = {"mel", "bcc", "akiec"}
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DESVIO = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LADO_REDE = 224

# taxa de aprendizado do backbone por arquitetura (a cabeça usa 10x mais)
LR_BACKBONE = {"efficientnet_b0": 3e-4, "resnet50": 1e-4, "vit_b_16": 3e-5, "convnext_tiny": 1e-4}


# ============================================================================ dados
def carregar_ham(metadata_csv, img_dirs):
    meta = pd.read_csv(metadata_csv)
    meta["is_malignant"] = meta.dx.isin(MALIGNAS).astype(int)
    caminho = {p.stem: str(p) for d in img_dirs for p in Path(d).glob("*.jpg")}
    meta["image_path"] = meta.image_id.map(caminho)
    falta = meta.image_path.isna().sum()
    if falta:
        print(f"aviso: {falta} imagens do CSV não foram encontradas nas pastas")
    return meta.dropna(subset=["image_path"]).reset_index(drop=True)


def split_por_lesao(meta, frac_val=0.10, frac_teste=0.25, seed=42):
    """Mesma lesão nunca aparece em dois conjuntos. Estratificado por maligno/benigno.
    Cria as colunas split ∈ {treino, val, teste} e metade ∈ {0, 1} (só no teste, também por lesão)."""
    n_partes = 20
    dobra = np.zeros(len(meta), int)
    sgkf = StratifiedGroupKFold(n_splits=n_partes, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(meta, meta.is_malignant, meta.lesion_id)):
        dobra[te] = k
    n_te, n_val = round(frac_teste * n_partes), round(frac_val * n_partes)
    meta = meta.copy()
    meta["split"] = np.where(dobra < n_te, "teste", np.where(dobra < n_te + n_val, "val", "treino"))
    meta["metade"] = -1
    t = meta.index[meta.split == "teste"]
    sg2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sg2.split(meta.loc[t], meta.loc[t, "is_malignant"], meta.loc[t, "lesion_id"])):
        meta.loc[t[te], "metade"] = k
    return meta


def carregar_cache(caminhos, lado=256, n_threads=8):
    """Todas as imagens em memória, redimensionadas para lado×lado (uint8). ~2 GB para o HAM10000 inteiro."""
    def um(c):
        im = Image.open(c).convert("RGB").resize((lado, lado), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
    t0 = time.time()
    with ThreadPoolExecutor(n_threads) as ex:
        ims = list(ex.map(um, list(caminhos)))
    cache = torch.stack(ims)
    print(f"cache: {len(cache)} imagens {lado}x{lado} em {time.time() - t0:.0f}s ({cache.numel() / 1e9:.1f} GB)")
    return cache


def preparar(xb, device):
    """uint8 (N,3,H,W) → float normalizado 224×224 no device."""
    x = xb.to(device, non_blocking=True).float().div_(255)
    if x.shape[-1] != LADO_REDE:
        x = F.interpolate(x, size=(LADO_REDE, LADO_REDE), mode="bilinear", antialias=True, align_corners=False)
    return (x - MEDIA.to(device)) / DESVIO.to(device)


def aumentar(x):
    """Aumento de dados na GPU, diferente para cada imagem: giro qualquer (lesão não tem 'cima'),
    espelho, zoom 0.8–1.0, pequeno deslocamento, brilho e contraste. x já normalizado."""
    n, dev = x.shape[0], x.device
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.pi
    esc = 0.8 + 0.2 * torch.rand(n, device=dev)
    esp = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    tx, ty = [(torch.rand(n, device=dev) * 2 - 1) * 0.08 for _ in range(2)]
    c, s = torch.cos(ang) * esc, torch.sin(ang) * esc
    theta = torch.stack([torch.stack([c * esp, -s, tx], 1), torch.stack([s * esp, c, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    brilho = (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4
    contraste = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4
    m = x.mean((1, 2, 3), keepdim=True)
    return (x - m) * contraste + m + brilho


def lotes(cache, idx, bs):
    idx = np.asarray(idx)
    for i in range(0, len(idx), bs):
        yield cache[torch.from_numpy(idx[i:i + bs])]


# ============================================================================ redes
class Rede(nn.Module):
    """Backbone (devolve um vetor por imagem) + a cabeça do BRACIS. Saída = logit z; P = sigmoid(z)."""

    def __init__(self, backbone, dim, pontos, nome):
        super().__init__()
        self.backbone, self.nome = backbone, nome
        self.cabeca = nn.Sequential(nn.Dropout(0.4), nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
        self._pontos = pontos          # lista de módulos (não registrada de novo: já estão dentro do backbone)

    @property
    def pontos(self):
        return self._pontos

    def forward(self, x):
        return self.cabeca(self.backbone(x)).view(-1)


def _base(nome, pretreinado, pesos_locais):
    fab = getattr(models, nome)
    if pesos_locais:
        m = fab(weights=None)
        m.load_state_dict(torch.load(pesos_locais, map_location="cpu"))
        return m
    if not pretreinado:
        return fab(weights=None)
    try:
        return fab(weights="DEFAULT")
    except Exception as e:
        raise RuntimeError(
            f"Não consegui baixar os pesos do ImageNet para {nome} ({e}).\n"
            "No Kaggle: Settings → Internet → On. Sem internet: anexe os arquivos .pth dos pesos do torchvision "
            "como dataset e informe o caminho em PESOS_LOCAIS.") from e


def criar_rede(nome, pretreinado=True, pesos_locais=None):
    m = _base(nome, pretreinado, pesos_locais)
    if nome == "efficientnet_b0":
        dim = m.classifier[-1].in_features; m.classifier = nn.Identity()
        pontos = [m.features[0]] + [b for est in list(m.features.children())[1:-1] for b in est.children()] + [m.features[-1]]
    elif nome == "resnet50":
        dim = m.fc.in_features; m.fc = nn.Identity()
        pontos = [m.maxpool] + [b for camada in (m.layer1, m.layer2, m.layer3, m.layer4) for b in camada.children()]
    elif nome == "vit_b_16":
        dim = m.heads.head.in_features; m.heads = nn.Identity()
        pontos = list(m.encoder.layers.children())
    elif nome == "convnext_tiny":
        dim = m.classifier[-1].in_features; m.classifier[-1] = nn.Identity()
        pontos = [m.features[0]] + [b for i in (1, 3, 5, 7) for b in m.features[i].children()]
    else:
        raise ValueError(f"arquitetura não suportada: {nome}. Opções: efficientnet_b0, resnet50, vit_b_16, convnext_tiny")
    return Rede(m, dim, pontos, nome)


# ============================================================================ treino
@torch.no_grad()
def prever(rede, cache, idx, device, bs=64):
    rede.eval(); out = []
    for xb in lotes(cache, idx, bs):
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            out.append(torch.sigmoid(rede(preparar(xb, device)).float()).cpu())
    return torch.cat(out).numpy()


def treinar(rede, cache, idx_tr, y_tr, idx_val, y_val, device, epocas=12, bs=32, lr_backbone=1e-4,
            paciencia=4, seed=42, log=print):
    """AdamW + OneCycle, precisão mista, aumento na GPU. Guarda o melhor estado pela AUROC na validação."""
    g = torch.Generator().manual_seed(seed)
    rede.to(device)
    y_tr = np.asarray(y_tr, np.float32)
    pos_w = torch.tensor(math.sqrt((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)), device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW([{"params": rede.backbone.parameters(), "lr": lr_backbone},
                             {"params": rede.cabeca.parameters(), "lr": lr_backbone * 10}], weight_decay=1e-4)
    passos = epocas * math.ceil(len(idx_tr) / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_backbone, lr_backbone * 10], total_steps=passos, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    idx_tr = np.asarray(idx_tr); hist, melhor, estado, sem_melhora = [], -1, None, 0
    for ep in range(epocas):
        rede.train(); t0 = time.time(); perm = torch.randperm(len(idx_tr), generator=g).numpy(); tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            x = aumentar(preparar(cache[torch.from_numpy(idx_tr[b])], device))
            y = torch.from_numpy(y_tr[b]).to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                perda = lossf(rede(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(perda).backward(); scaler.step(opt); scaler.update(); sch.step()
            tot += perda.item() * len(b)
        p_val = prever(rede, cache, idx_val, device)
        auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) == 2 else float("nan")
        hist.append({"época": ep + 1, "perda treino": tot / len(perm), "AUROC val": auc, "segundos": time.time() - t0})
        log(f"  {rede.nome} época {ep + 1}/{epocas}: perda {tot / len(perm):.4f} | AUROC val {auc:.4f} | {time.time() - t0:.0f}s")
        if auc > melhor:
            melhor, sem_melhora = auc, 0
            estado = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                log(f"  parada antecipada (sem melhora há {paciencia} épocas)"); break
    if estado is not None:
        rede.load_state_dict(estado)
    return rede.eval(), pd.DataFrame(hist)


def escolher_limiar(p, y):
    """Limiar que maximiza a acurácia balanceada (Youden) na validação."""
    p, y = np.asarray(p), np.asarray(y)
    cands = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    ba = [0.5 * (((p >= t) & (y == 1)).sum() / max(y.sum(), 1) + ((p < t) & (y == 0)).sum() / max((1 - y).sum(), 1))
          for t in cands]
    return float(cands[int(np.argmax(ba))])
