"""
dados.py — carrega cada conjunto de dados do mesmo jeito: treino, (validação), teste, rótulos e "vistas" (TTA).

HAM10000: uma imagem por lesão, N lesões sorteadas pela semente, divididas em treino/validação/teste
          estratificado; nenhuma lesão aparece em dois conjuntos. As outras fotos das lesões de teste ficam
          guardadas para a análise "a rede muda de ideia entre fotos da mesma lesão?".
CIFAR/SVHN/MNIST: divisões oficiais do torchvision (baixados automaticamente). Ruído: uma fração dos rótulos
          de TREINO é trocada por outra classe ao acaso (o teste continua limpo).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import config as C

NORMAS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
          "cifar100": ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
          "svhn": ((0.438, 0.444, 0.473), (0.198, 0.201, 0.197)),
          "mnist": ((0.1307,), (0.3081,)),
          "ham10000": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}
N_CLASSES = {"cifar10": 10, "cifar100": 100, "svhn": 10, "mnist": 10, "ham10000": 2}


# ----------------------------------------------------------------------------- vistas (TTA), aplicadas no tensor já normalizado
def _desloca(dx, dy):
    def f(x):
        p = 2
        x = F.pad(x, (p, p, p, p), mode="replicate")
        h, w = x.shape[2] - 2 * p, x.shape[3] - 2 * p
        return x[:, :, p + dy: p + dy + h, p + dx: p + dx + w]
    return f


VISTAS = {
    "ham10000": {"original": lambda x: x, "espelho_h": lambda x: torch.flip(x, [3]),
                 "espelho_v": lambda x: torch.flip(x, [2]), "giro_180": lambda x: torch.flip(x, [2, 3])},
    "cifar10": {"original": lambda x: x, "espelho_h": lambda x: torch.flip(x, [3]),
                "desloca_+2": _desloca(2, 2), "desloca_-2": _desloca(-2, -2)},
    "svhn": {"original": lambda x: x, "desloca_dir": _desloca(2, 0), "desloca_esq": _desloca(-2, 0), "desloca_baixo": _desloca(0, 2)},
    "mnist": {"original": lambda x: x, "desloca_dir": _desloca(1, 0), "desloca_esq": _desloca(-1, 0), "desloca_baixo": _desloca(0, 1)},
}
VISTAS["cifar100"] = VISTAS["cifar10"]


# ----------------------------------------------------------------------------- utilidades
class ComRotulos(Dataset):
    """Embrulha um dataset e devolve (imagem, rótulo escolhido por nós, índice)."""

    def __init__(self, base, rotulos, indices=None):
        self.base, self.rotulos = base, np.asarray(rotulos)
        self.indices = np.arange(len(base)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return self.base[j][0], int(self.rotulos[j]), i


class Imagens(Dataset):
    """Imagens em disco (HAM10000)."""

    def __init__(self, caminhos, rotulos, tfm):
        self.c, self.y, self.t = list(caminhos), np.asarray(rotulos), tfm

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        return self.t(Image.open(self.c[i]).convert("RGB")), int(self.y[i]), i


@dataclass
class Conjunto:
    nome: str
    n_classes: int
    treino: Dataset                 # com aumento de dados (para treinar)
    treino_ref: Dataset             # sem aumento, subconjunto de referência (probes, vizinhos, Mahalanobis)
    teste: Dataset                  # sem aumento
    y_ref: np.ndarray               # rótulos de treino (com ruído, se houver) da referência
    y_teste: np.ndarray             # rótulos verdadeiros do teste
    val: Dataset = None
    y_val: np.ndarray = None
    meta_teste: pd.DataFrame = None
    vistas: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    @property
    def binario(self):
        return self.n_classes == 2


def _rotulos(ds):
    for a in ("targets", "labels"):
        if hasattr(ds, a):
            return np.asarray(getattr(ds, a))
    return np.array([ds[i][1] for i in range(len(ds))])


def _ruido(y, frac, n_classes, rng):
    y = y.copy()
    trocar = rng.random(len(y)) < frac
    y[trocar] = (y[trocar] + rng.integers(1, n_classes, trocar.sum())) % n_classes   # sempre outra classe
    return y


# ----------------------------------------------------------------------------- conjuntos
def _torchvision(nome, semente, ruido):
    media, desvio = NORMAS[nome]
    norm = transforms.Normalize(media, desvio)
    k = N_CLASSES[nome]
    if nome == "mnist":
        aug = transforms.Compose([transforms.RandomCrop(28, padding=2), transforms.ToTensor(), norm])
    elif nome == "svhn":
        aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor(), norm])
    else:
        aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    simples = transforms.Compose([transforms.ToTensor(), norm])

    if C.TESTE_RAPIDO:                                    # dados falsos, só para testar o código
        forma = (1, 28, 28) if nome == "mnist" else (3, 32, 32)
        mk = lambda n, t, off: datasets.FakeData(n, forma, k, transform=t, random_offset=off)
        tr_aug, tr, te = mk(600, aug, 0), mk(600, simples, 0), mk(300, simples, 10_000)
    else:
        raiz = str(C.DADOS)
        if nome == "svhn":
            mk = lambda split, t: datasets.SVHN(raiz, split=split, download=True, transform=t)
            tr_aug, tr, te = mk("train", aug), mk("train", simples), mk("test", simples)
        else:
            cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100, "mnist": datasets.MNIST}[nome]
            mk = lambda train, t: cls(raiz, train=train, download=True, transform=t)
            tr_aug, tr, te = mk(True, aug), mk(True, simples), mk(False, simples)

    rng = np.random.default_rng(semente)
    y_tr = _rotulos(tr)
    if ruido > 0:
        y_tr = _ruido(y_tr, ruido, k, rng)
    y_te = _rotulos(te)
    ref = rng.choice(len(tr), size=min(C.N_REF, len(tr)), replace=False)
    meta = pd.DataFrame({"indice": np.arange(len(te)), "rotulo": y_te})
    return Conjunto(nome, k, ComRotulos(tr_aug, y_tr), ComRotulos(tr, y_tr, ref), ComRotulos(te, y_te),
                    y_ref=y_tr[ref], y_teste=y_te, meta_teste=meta, vistas=VISTAS[nome])


def _ham(semente):
    from sklearn.model_selection import train_test_split
    meta = pd.read_csv(C.HAM_DIR / "HAM10000_metadata.csv")
    meta["is_malignant"] = meta.dx.isin({"mel", "bcc", "akiec"}).astype(int)
    caminho = {a.stem: str(a) for a in Path(C.HAM_DIR).rglob("*.jpg")}
    meta["image_path"] = meta.image_id.map(caminho)
    meta = meta.dropna(subset=["image_path"])

    uma = meta.sample(frac=1, random_state=semente).drop_duplicates("lesion_id").reset_index(drop=True)
    n = min(C.HAM_N_LESOES or len(uma), len(uma))
    base = uma if n == len(uma) else train_test_split(uma, train_size=n, stratify=uma.is_malignant, random_state=semente)[0]
    tr, resto = train_test_split(base, test_size=C.HAM_FRAC_VAL + C.HAM_FRAC_TESTE, stratify=base.is_malignant, random_state=semente)
    va, te = train_test_split(resto, test_size=C.HAM_FRAC_TESTE / (C.HAM_FRAC_VAL + C.HAM_FRAC_TESTE),
                              stratify=resto.is_malignant, random_state=semente)
    tr, va, te = [d.reset_index(drop=True) for d in (tr, va, te)]
    for a, b in [(tr, va), (tr, te), (va, te)]:
        assert not set(a.lesion_id) & set(b.lesion_id), "há lesão em dois conjuntos"

    media, desvio = NORMAS["ham10000"]
    norm = transforms.Normalize(media, desvio)
    T = 64 if C.TESTE_RAPIDO else C.HAM_TAM
    simples = transforms.Compose([transforms.Resize((T, T)), transforms.ToTensor(), norm])
    aug = transforms.Compose([transforms.Resize((T, T)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                              transforms.RandomRotation(20), transforms.ColorJitter(0.1, 0.1, 0.1), transforms.ToTensor(), norm])
    ref = tr.sample(n=min(C.N_REF, len(tr)), random_state=semente).reset_index(drop=True)
    extras = meta[meta.lesion_id.isin(te.lesion_id) & ~meta.image_id.isin(te.image_id)].reset_index(drop=True)
    return Conjunto("ham10000", 2, Imagens(tr.image_path, tr.is_malignant, aug), Imagens(ref.image_path, ref.is_malignant, simples),
                    Imagens(te.image_path, te.is_malignant, simples), y_ref=ref.is_malignant.values, y_teste=te.is_malignant.values,
                    val=Imagens(va.image_path, va.is_malignant, simples), y_val=va.is_malignant.values,
                    meta_teste=te[["image_id", "lesion_id", "dx", "dx_type", "is_malignant", "image_path"]].rename(columns={"is_malignant": "rotulo"}),
                    vistas=VISTAS["ham10000"],
                    extras={"fotos_extras": Imagens(extras.image_path, extras.is_malignant, simples),
                            "fotos_extras_meta": extras[["image_id", "lesion_id"]],
                            "divisao": pd.concat([d.assign(split=s) for d, s in [(tr, "treino"), (va, "validacao"), (te, "teste")]])
                                       [["image_id", "lesion_id", "dx", "dx_type", "is_malignant", "split"]]})


def carregar(exp, semente):
    e = C.EXPERIMENTOS[exp]
    return _ham(semente) if e["dados"] == "ham10000" else _torchvision(e["dados"], semente, e["ruido"])


def carregador(ds, lote, embaralhar=False):
    return torch.utils.data.DataLoader(ds, batch_size=lote, shuffle=embaralhar, num_workers=C.TRABALHADORES,
                                       pin_memory=torch.cuda.is_available(), persistent_workers=False)
