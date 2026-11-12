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
NORMAS["animal10n"] = NORMAS["chaoyang"] = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   # médias do ImageNet
# tons de cinza levados a 32×32 e 3 canais: as mesmas redes da grade (ResNet-32, VGG-16, MobileNetV2, CNN) servem sem mudança
NORMAS["mnist32"] = ((0.1307,) * 3, (0.3081,) * 3)
NORMAS["fashion32"] = ((0.2860,) * 3, (0.3530,) * 3)
N_CLASSES = {"cifar10": 10, "cifar100": 100, "svhn": 10, "mnist": 10, "ham10000": 2, "animal10n": 10, "chaoyang": 4,
             "mnist32": 10, "fashion32": 10}


# ----------------------------------------------------------------------------- vistas (TTA), aplicadas no tensor já normalizado
def _desloca(dx, dy):
    def f(x):
        p = max(2, abs(dx), abs(dy))          # a borda precisa cobrir o deslocamento
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
VISTAS["animal10n"] = {"original": lambda x: x, "espelho_h": lambda x: torch.flip(x, [3]),
                       "desloca_+4": _desloca(4, 4), "desloca_-4": _desloca(-4, -4)}      # 64×64: deslocamento proporcional
VISTAS["chaoyang"] = VISTAS["ham10000"]                                                  # lâmina: espelhos e giro não mudam a classe
VISTAS["fashion32"] = VISTAS["cifar10"]                                                  # roupa espelhada continua a mesma peça
VISTAS["mnist32"] = VISTAS["mnist"]                                                      # dígitos: só deslocamentos (sem espelho)


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
    """Imagens em disco (HAM10000, ANIMAL-10N, Chaoyang)."""

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
    epocas: int = None              # se o experimento pedir um número de épocas próprio

    @property
    def binario(self):
        return self.n_classes == 2


def _rotulos(ds):
    for a in ("targets", "labels"):
        if hasattr(ds, a):
            return np.asarray(getattr(ds, a))
    return np.array([ds[i][1] for i in range(len(ds))])


CIFARN_URL = "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/{}"
CIFARN_ARQ = {"cifar10": "CIFAR-10_human.pt", "cifar100": "CIFAR-100_human.pt"}


def _rotulos_humanos(nome, chave, y_limpo):
    """Rótulos de treino dados por PESSOAS (CIFAR-10N / CIFAR-100N; Wei et al., ICLR 2022): os erros humanos reais.
    chave: aggre_label (~9% errados), random_label1 (~17%), worse_label (~40%) no CIFAR-10N; noisy_label (~40%) no CIFAR-100N.
    Baixa o arquivo do repositório dos autores na primeira vez (dados/cifar-n/)."""
    import torch, urllib.request
    pasta = Path(C.DADOS) / "cifar-n"; pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / CIFARN_ARQ[nome]
    if not arq.exists():
        print(f"  baixando {arq.name} (rótulos humanos)...", flush=True)
        urllib.request.urlretrieve(CIFARN_URL.format(arq.name), arq)
    r = torch.load(arq, weights_only=False)          # dicionário de vetores numpy, do repositório oficial do CIFAR-N
    limpo = np.asarray(r["clean_label"])
    assert len(limpo) == len(y_limpo) and (limpo == y_limpo).all(), "a ordem do CIFAR-N não bate com a do torchvision"
    y = np.asarray(r[chave]).astype(np.int64)
    print(f"  CIFAR-N '{chave}': {100 * (y != limpo).mean():.1f}% dos rótulos de treino errados", flush=True)
    return y


def _ruido(y, frac, n_classes, rng):
    y = y.copy()
    trocar = rng.random(len(y)) < frac
    y[trocar] = (y[trocar] + rng.integers(1, n_classes, trocar.sum())) % n_classes   # sempre outra classe
    return y


# ----------------------------------------------------------------------------- conjuntos
def _torchvision(nome, semente, ruido, epocas=None):
    media, desvio = NORMAS[nome]
    norm = transforms.Normalize(media, desvio)
    k = N_CLASSES[nome]
    cinza32 = nome in ("mnist32", "fashion32")                   # 28×28 cinza -> 32×32 com 3 canais
    pre = [transforms.Pad(2), transforms.Grayscale(3)] if cinza32 else []
    if nome == "mnist":
        aug = transforms.Compose([transforms.RandomCrop(28, padding=2), transforms.ToTensor(), norm])
    elif nome == "mnist32":
        aug = transforms.Compose(pre + [transforms.RandomCrop(32, padding=2), transforms.ToTensor(), norm])
    elif nome == "fashion32":
        aug = transforms.Compose(pre + [transforms.RandomCrop(32, padding=2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    elif nome == "svhn":
        aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor(), norm])
    else:
        aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    simples = transforms.Compose(pre + [transforms.ToTensor(), norm])

    if C.TESTE_RAPIDO:                                    # dados falsos, só para testar o código
        forma = (1, 28, 28) if nome in ("mnist", "mnist32", "fashion32") else (3, 32, 32)
        mk = lambda n, t, off: datasets.FakeData(n, forma, k, transform=t, random_offset=off)
        tr_aug, tr, te = mk(600, aug, 0), mk(600, simples, 0), mk(300, simples, 10_000)
    else:
        raiz = str(C.DADOS)
        if nome == "svhn":
            mk = lambda split, t: datasets.SVHN(raiz, split=split, download=True, transform=t)
            tr_aug, tr, te = mk("train", aug), mk("train", simples), mk("test", simples)
        else:
            cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100, "mnist": datasets.MNIST, "mnist32": datasets.MNIST,
                   "fashion32": datasets.FashionMNIST}[nome]
            mk = lambda train, t: cls(raiz, train=train, download=True, transform=t)
            tr_aug, tr, te = mk(True, aug), mk(True, simples), mk(False, simples)

    rng = np.random.default_rng(semente)
    y_tr = _rotulos(tr)
    if isinstance(ruido, str) and ruido.startswith("humano:"):       # CIFAR-N: rótulos de treino dados por pessoas
        if not C.TESTE_RAPIDO:
            y_tr = _rotulos_humanos(nome, ruido.split(":", 1)[1], y_tr)
        else:                                                           # teste rápido: imita com ruído sintético
            y_tr = _ruido(y_tr, 0.2, k, rng)
    elif ruido > 0:
        y_tr = _ruido(y_tr, ruido, k, rng)
    y_te = _rotulos(te)
    ref = rng.choice(len(tr), size=min(C.N_REF, len(tr)), replace=False)
    meta = pd.DataFrame({"indice": np.arange(len(te)), "rotulo": y_te})
    return Conjunto(nome, k, ComRotulos(tr_aug, y_tr), ComRotulos(tr, y_tr, ref), ComRotulos(te, y_te),
                    y_ref=y_tr[ref], y_teste=y_te, meta_teste=meta, vistas=VISTAS[nome], epocas=epocas)


def _ham(semente, n_lesoes):
    from sklearn.model_selection import train_test_split
    meta = pd.read_csv(C.HAM_DIR / "HAM10000_metadata.csv")
    meta["is_malignant"] = meta.dx.isin({"mel", "bcc", "akiec"}).astype(int)
    caminho = {a.stem: str(a) for a in Path(C.HAM_DIR).rglob("*.jpg")}
    meta["image_path"] = meta.image_id.map(caminho)
    meta = meta.dropna(subset=["image_path"])

    uma = meta.sample(frac=1, random_state=semente).drop_duplicates("lesion_id").reset_index(drop=True)
    n = min(n_lesoes or len(uma), len(uma))
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


def _pastas_imagens(raiz, nome_ds, ajuda):
    if not Path(raiz).exists():
        raise FileNotFoundError(f"{nome_ds}: não achei {raiz}/. {ajuda}")


def _animal10n(semente, epocas=None):
    """ANIMAL-10N (Song et al., ICML 2019): 50.000 imagens de treino com ~8% de rótulos errados (vindos da web)
    e 5.000 de teste verificadas. Arquivos "<classe>_<n>.jpg" dentro de pastas de treino e de teste."""
    raiz = C.ANIMAL_DIR
    _pastas_imagens(raiz, "ANIMAL-10N", "Descompacte lá o arquivo do dataset (pastas training/ e testing/).")
    arqs = [p for p in Path(raiz).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    def split(p):
        s = "/".join(p.parts).lower()
        return "teste" if "test" in s else ("treino" if "train" in s else None)
    tab = pd.DataFrame({"image_path": [str(p) for p in arqs], "split": [split(p) for p in arqs],
                        "rotulo": [int(p.stem.split("_")[0]) for p in arqs]}).dropna()
    tr, te = tab[tab.split == "treino"].reset_index(drop=True), tab[tab.split == "teste"].reset_index(drop=True)
    if not C.TESTE_RAPIDO and (len(tr) < 45000 or len(te) < 4500 or tab.rotulo.nunique() != 10):
        raise ValueError(f"ANIMAL-10N: esperava ~50.000 treino / 5.000 teste / 10 classes; achei {len(tr)} / {len(te)} / "
                         f"{tab.rotulo.nunique()}. Confira o unzip em {raiz}/.")
    media, desvio = NORMAS["animal10n"]; norm = transforms.Normalize(media, desvio)
    simples = transforms.Compose([transforms.ToTensor(), norm])
    aug = transforms.Compose([transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    rng = np.random.default_rng(semente)
    ref = rng.choice(len(tr), size=min(C.N_REF, len(tr)), replace=False)
    refs = tr.iloc[ref].reset_index(drop=True)
    meta = pd.DataFrame({"indice": np.arange(len(te)), "rotulo": te.rotulo.values})
    return Conjunto("animal10n", 10, Imagens(tr.image_path, tr.rotulo, aug), Imagens(refs.image_path, refs.rotulo, simples),
                    Imagens(te.image_path, te.rotulo, simples), y_ref=refs.rotulo.values, y_teste=te.rotulo.values,
                    meta_teste=meta, vistas=VISTAS["animal10n"], epocas=epocas)


def _chaoyang(semente, arq="effnet_b0"):
    """Chaoyang (Zhu et al., IEEE TMI 2022): histopatologia colorretal, 4 classes (normal, serrada, adenocarcinoma, adenoma).
    Rótulos de treino com ruído (discordância entre patologistas); teste com rótulos de consenso.
    Lê train.json e test.json: listas de {"name": caminho relativo ao json, "label": classe}."""
    import json
    from sklearn.model_selection import train_test_split
    raiz = C.CHAOYANG_DIR
    _pastas_imagens(raiz, "Chaoyang", "Descompacte lá o dataset (com train.json, test.json e as imagens).")
    def ler(nome):
        js = next(iter(Path(raiz).rglob(nome)), None)
        if js is None:
            raise FileNotFoundError(f"Chaoyang: não achei {nome} dentro de {raiz}/.")
        t = pd.DataFrame(json.load(open(js)))
        t["image_path"] = [str(js.parent / n) for n in t["name"]]
        return t.rename(columns={"label": "rotulo"})[["image_path", "rotulo"]]
    tr_all, te = ler("train.json"), ler("test.json")
    tr, va = train_test_split(tr_all, test_size=0.1, stratify=tr_all.rotulo, random_state=semente)   # validação: escolher a época
    tr, va, te = [x.reset_index(drop=True) for x in (tr, va, te)]
    media, desvio = NORMAS["chaoyang"]; norm = transforms.Normalize(media, desvio)
    T = 224 if (not C.TESTE_RAPIDO or arq == "vit_b_16") else 64     # o ViT só aceita 224×224
    simples = transforms.Compose([transforms.Resize((T, T)), transforms.ToTensor(), norm])
    aug = transforms.Compose([transforms.Resize((T, T)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                              transforms.RandomRotation(20), transforms.ColorJitter(0.1, 0.1, 0.1), transforms.ToTensor(), norm])
    ref = tr.sample(n=min(C.N_REF, len(tr)), random_state=semente).reset_index(drop=True)
    meta = pd.DataFrame({"indice": np.arange(len(te)), "rotulo": te.rotulo.values})
    return Conjunto("chaoyang", 4, Imagens(tr.image_path, tr.rotulo, aug), Imagens(ref.image_path, ref.rotulo, simples),
                    Imagens(te.image_path, te.rotulo, simples), y_ref=ref.rotulo.values, y_teste=te.rotulo.values,
                    val=Imagens(va.image_path, va.rotulo, simples), y_val=va.rotulo.values,
                    meta_teste=meta, vistas=VISTAS["chaoyang"])


def carregar(exp, semente):
    e = C.EXPERIMENTOS[exp]
    if e["dados"] == "ham10000":
        return _ham(semente, e.get("n_lesoes", C.HAM_N_LESOES))
    if e["dados"] == "animal10n":
        return _animal10n(semente, e.get("epocas"))
    if e["dados"] == "chaoyang":
        return _chaoyang(semente, e["arq"])
    return _torchvision(e["dados"], semente, e["ruido"], e.get("epocas"))


def carregador(ds, lote, embaralhar=False):
    return torch.utils.data.DataLoader(ds, batch_size=lote, shuffle=embaralhar, num_workers=C.TRABALHADORES,
                                       pin_memory=torch.cuda.is_available(), persistent_workers=False)
