"""
classicos.py — protocolo de Chen et al. (AISTATS 2019) em três datasets clássicos (CIFAR-10, CIFAR-100, Tiny-ImageNet)
e comparação com o nosso método (dinâmica interna + regras fuzzy). Só o CIFAR-10 tem números publicados por Chen.
O protocolo completo, com o que é igual e o que difere de Chen, está em PROTOCOLO.md na raiz do repositório.

Protocolo de Chen (seção 5 do artigo):
  treino do CIFAR-10 (50k) -> TRAIN-BASE 30k (treina a ResNet) | TRAIN-META 10k (treina probes e meta-modelos) | DEV 10k
  TEST = teste do CIFAR-10 (10k, sempre limpo) | OOD = TEST + teste do CIFAR-100 (todo OOD conta como "erro")
  Cenário ruidoso: 30% dos rótulos trocados por outra classe aleatória nos conjuntos de treino.
  Alvo do meta-modelo: a ResNet acertou (ou errou) aquela imagem.
Métodos de Chen: Softmax | Blackbox LR/GBM (vetor de saída) | Probes LR/GBM (saídas de todos os probes concatenadas)
Nosso: features de dinâmica (trocas_opiniao, prof_decisao, ...) -> regressão logística, fuzzy aditivo e árvore fuzzy.
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-9
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
NOSSAS = ["trocas_opiniao", "prof_decisao", "entropia_probes", "conf_fim", "salto_final",
          "vistas_mudam", "p_vistas_std", "trajetoria_vistas", "vizinhos_discordam", "dist_vizinhos",
          "choquet_fna", "prox_p"]


# ----------------------------------------------------------------------------- dados
def dividir(y_treino, seed=0):
    """Índices TRAIN-BASE (30k), TRAIN-META (10k), DEV (10k), proporcionais se o conjunto for menor."""
    n = len(y_treino); p = np.random.default_rng(seed).permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return p[:a], p[a:b], p[b:]


def ruido(y, frac, n_classes=10, seed=1):
    """Troca o rótulo de uma fração das amostras por OUTRA classe aleatória."""
    y = y.copy(); r = np.random.default_rng(seed)
    idx = r.choice(len(y), int(round(frac * len(y))), replace=False)
    y[idx] = (y[idx] + r.integers(1, n_classes, len(idx))) % n_classes
    return y


def tensor(X):  # uint8 (N,32,32,3) -> float normalizado (N,3,32,32)
    return (torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD


def aumentar(x):  # recorte aleatório com padding de 1/8 do lado + espelho horizontal, no lote
    n, L = x.shape[0], x.shape[-1]; pad = L // 8; xp = Fn.pad(x, (pad, pad, pad, pad), mode="reflect")
    i, j = np.random.randint(0, 2 * pad + 1, 2)
    x = xp[:, :, i:i + L, j:j + L]
    m = torch.rand(n, device=x.device) < 0.5
    x[m] = x[m].flip(3)
    return x


# ----------------------------------------------------------------------------- ResNet-32 (CIFAR, He et al. 2016)
class Bloco(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.atalho = nn.Sequential() if stride == 1 and cin == cout else \
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return Fn.relu(self.b2(self.c2(Fn.relu(self.b1(self.c1(x))))) + self.atalho(x))


class ResNetCifar(nn.Module):
    """ResNet-32: entrada + 15 unidades residuais (= 16 pontos de leitura, como as ~17 camadas de Chen)."""

    def __init__(self, n=5, n_classes=10):
        super().__init__()
        self.entrada = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        blocos, c = [], 16
        for cout, s in [(16, 1), (32, 2), (64, 2)]:
            for k in range(n):
                blocos.append(Bloco(c, cout, s if k == 0 else 1)); c = cout
        self.unidades = nn.Sequential(*blocos)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.fc(self.unidades(self.entrada(x)).mean((2, 3)))

    def pontos_de_leitura(self):
        return [self.entrada] + list(self.unidades)


# ----------------------------------------------------------------------------- a rede de Chen (o mais fiel possível)
class UnidadePre(nn.Module):
    """Unidade residual pré-ativação com ReLU 'vazado' (0,1), como no exemplo ResNet do TensorFlow citado por Chen."""

    def __init__(self, cin, cout, stride, ativar_antes=False):
        super().__init__()
        self.b1 = nn.BatchNorm2d(cin); self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout); self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.ativar_antes, self.stride, self.cin, self.cout = ativar_antes, stride, cin, cout

    def _atalho(self, x):
        if self.stride > 1:
            x = Fn.avg_pool2d(x, self.stride, self.stride)
        if self.cin != self.cout:                                   # completa os canais com zeros (sem parâmetros)
            p = (self.cout - self.cin) // 2
            x = Fn.pad(x, (0, 0, 0, 0, p, self.cout - self.cin - p))
        return x

    def forward(self, x):
        a = Fn.leaky_relu(self.b1(x), 0.1)
        orig = a if self.ativar_antes else x
        h = self.c2(Fn.leaky_relu(self.b2(self.c1(a)), 0.1))
        return h + self._atalho(orig)


class Ultima(nn.Module):
    """'unit_last': BN + ReLU vazado antes do pooling global."""

    def __init__(self, c):
        super().__init__(); self.b = nn.BatchNorm2d(c)

    def forward(self, x):
        return Fn.leaky_relu(self.b(x), 0.1)


class ResNetChen(nn.Module):
    """ResNet-32 pré-ativação do exemplo do TensorFlow (Chen et al. 2019, seção 5): filtros 16/16/32/64,
    5 unidades por estágio, ReLU vazado 0,1, padronização por imagem.
    Pontos de leitura = convolução inicial + 15 unidades + unit_last = 17 (as '17 camadas' de Chen)."""

    def __init__(self, n=5, n_classes=10):
        super().__init__()
        self.entrada = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        unidades, c = [], 16
        for k_est, (cout, s) in enumerate([(16, 1), (32, 2), (64, 2)]):
            for k in range(n):
                unidades.append(UnidadePre(c, cout, s if k == 0 else 1, ativar_antes=(k_est == 0 and k == 0))); c = cout
        self.unidades = nn.Sequential(*unidades)
        self.ultima = Ultima(64)
        self.fc = nn.Linear(64, n_classes)

    @staticmethod
    def padronizar(x):
        """tf.image.per_image_standardization: média 0 e desvio 1 em cada imagem."""
        m = x.mean((1, 2, 3), keepdim=True); d = x.std((1, 2, 3), keepdim=True)
        return (x - m) / torch.clamp(d, min=1.0 / np.sqrt(x[0].numel()))

    def forward(self, x):
        return self.fc(self.ultima(self.unidades(self.entrada(self.padronizar(x)))).mean((2, 3)))

    def pontos_de_leitura(self):
        return [self.entrada] + list(self.unidades) + [self.ultima]


def aumentar_por_imagem(x):
    """Recorte aleatório (padding de 1/8 do lado, com zeros: 4 px no CIFAR) e espelho horizontal, por imagem."""
    n, dev, L = x.shape[0], x.device, x.shape[-1]; pad = L // 8
    xp = Fn.pad(x, (pad, pad, pad, pad))
    i, j = torch.randint(0, 2 * pad + 1, (n,), device=dev), torch.randint(0, 2 * pad + 1, (n,), device=dev)
    ar = torch.arange(L, device=dev)
    lin, col = (i[:, None] + ar)[:, None, :, None], (j[:, None] + ar)[:, None, None, :]
    x = xp[torch.arange(n, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None], lin, col]
    m = torch.rand(n, device=dev) < 0.5
    x[m] = x[m].flip(3)
    return x


def treinar_chen(model, X, y, Xval, yval, device, passos=90_000, bs=128, log=print):
    """Receita do exemplo do TensorFlow: SGD com momento 0,9, lote 128, decaimento de peso 2e-4,
    taxa 0,1 → 0,01 (40k passos) → 0,001 (60k) → 0,0001 (80k), em 90k passos.
    Com outro nº de passos, as quedas da taxa ficam nas mesmas frações (4/9, 6/9, 8/9 do treino)."""
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, [round(passos * f) for f in (4 / 9, 6 / 9, 8 / 9)], 0.1)
    Xt, yt = tensor(X), torch.from_numpy(y).long()
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    passo, t0 = 0, time.time()
    while passo < passos:
        model.train(); p = torch.randperm(len(Xt))
        for i in range(0, len(p) - bs + 1, bs):
            b = p[i:i + bs]; xb = aumentar_por_imagem(Xt[b].to(device)); yb = yt[b].to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = Fn.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            passo += 1
            if passo == 1_000 and passos > 1_000:
                log(f"  passo 1000: {(time.time() - t0) / 60:.1f} min → previsão para {passos} passos: "
                    f"{(time.time() - t0) / 60 * passos / 1000:.0f} min"); model.train()
            if passo % 10_000 == 0 or passo == passos:
                log(f"  passo {passo}/{passos}  perda {loss.item():.3f}  acurácia DEV {acuracia(model, Xval, yval, device):.3f}"
                    f"  ({(time.time() - t0) / 60:.0f} min)"); model.train()
            if passo >= passos:
                break
    return model


def treinar(model, X, y, Xval, yval, epocas, device, bs=128, lr=0.1, log=print):
    model.to(device); opt = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    passos = epocas * int(np.ceil(len(X) / bs)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=passos, pct_start=0.15)
    Xt, yt = tensor(X), torch.from_numpy(y).long()
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    for ep in range(epocas):
        model.train(); t0 = time.time(); p = torch.randperm(len(Xt))
        for i in range(0, len(p), bs):
            b = p[i:i + bs]; xb = aumentar(Xt[b].to(device)); yb = yt[b].to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = Fn.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        if ep % 5 == 4 or ep == epocas - 1:
            log(f"  época {ep + 1}/{epocas}  perda {loss.item():.3f}  acurácia DEV {acuracia(model, Xval, yval, device):.3f}  ({time.time() - t0:.0f}s/época)")
    return model


@torch.no_grad()
def acuracia(model, X, y, device, bs=500):
    model.eval(); Xt = tensor(X)
    pred = torch.cat([model(Xt[i:i + bs].to(device)).argmax(1).cpu() for i in range(0, len(Xt), bs)]).numpy()
    return (pred == y).mean()


# ----------------------------------------------------------------------------- leitura da rede
def _desloca(x, dx, dy):
    """Desloca a imagem dx px para a direita e dy px para baixo (borda repetida)."""
    H, W = x.shape[2:]
    return Fn.pad(x, (dx, 0, dy, 0), mode="replicate")[:, :, :H, :W]


VISTAS = {"original": lambda x: x, "espelho_h": lambda x: x.flip(3),
          "desloca_dir": lambda x: _desloca(x, 2, 0), "desloca_baixo": lambda x: _desloca(x, 0, 2)}


@torch.no_grad()
def ler_ativacoes(model, X, device, vista=lambda x: x, bs=500):
    """Ativações médias de cada ponto de leitura + probabilidades da saída."""
    model.eval(); pts = model.pontos_de_leitura(); B = len(pts)
    acum, probs = [[] for _ in range(B)], []
    Xt = tensor(X)
    for i in range(0, len(Xt), bs):
        cap = [None] * B
        hs = [m.register_forward_hook(lambda m, a, o, k=k: cap.__setitem__(k, o.float().mean((2, 3)).cpu().numpy())) for k, m in enumerate(pts)]
        try:
            probs.append(torch.softmax(model(vista(Xt[i:i + bs].to(device))).float(), 1).cpu().numpy())
        finally:
            for h in hs: h.remove()
        for k in range(B): acum[k].append(cap[k])
    return [np.concatenate(a).astype(np.float32) for a in acum], np.concatenate(probs)


def ler_tudo(model, X, device):
    return {v: ler_ativacoes(model, X, device, f) for v, f in VISTAS.items()}


class Probes:
    """Um classificador linear por ponto de leitura. alvo = rótulos (Chen) ou decisão da rede (nosso)."""

    def fit(self, A, alvo, n_classes, C=0.1):
        self.K = n_classes
        from sklearn.dummy import DummyClassifier
        fab = (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))) if len(np.unique(alvo)) > 1 \
            else DummyClassifier                                                     # rede que só prevê uma classe (só em testes)
        self.m = [fab().fit(a, alvo) for a in A]
        self.classes = self.m[0].classes_
        return self

    def proba(self, A):  # (n, B, K) com K = nº de classes (completa classes ausentes com 0)
        out = np.zeros((len(A[0]), len(A), self.K), np.float32)
        for k, (m, a) in enumerate(zip(self.m, A)):
            out[:, k, m.classes_] = m.predict_proba(a)
        return out


def choquet_fna(C):
    """Integral C_F de profundidade com F_NA e capacidade nu1 (seção de Choquet de profundidade)."""
    n, w = C.shape; tt = np.arange(1, w + 1, dtype=float)
    o = np.argsort(C, 1); Cs = np.take_along_axis(C, o, 1)
    nu = np.cumsum(tt[o][:, ::-1], 1)[:, ::-1] / tt.sum()
    x = np.diff(np.concatenate([np.zeros((n, 1)), Cs], 1), axis=1)
    return np.minimum(1, np.where(x <= nu, x, np.minimum(x / 2, nu)).sum(1))


def features_nossas(leit, probes, banco_U, banco_dec, dist_ref, k=15, excluir_si=False):
    """Mesmas features do notebook do HAM10000, generalizadas para várias classes."""
    A, Pv = leit["original"]
    yhat = Pv.argmax(1); n = len(yhat); B = len(A)
    pp = probes.proba(A)                                   # (n, B, K)
    conf = pp[np.arange(n), :, yhat]                       # chance de cada camada já dar a resposta final
    op = pp.argmax(2) == yhat[:, None]                     # a camada concorda com a resposta final?
    f = {}
    f["trocas_opiniao"] = np.abs(np.diff(op.astype(int), axis=1)).sum(1)
    f["prof_decisao"] = np.where(~op, np.arange(B)[None] + 1, 0).max(1) / B
    f["entropia_probes"] = -(pp * np.log(pp + EPS)).sum(2).mean(1) / np.log(pp.shape[2])
    f["conf_fim"] = conf[:, -max(1, B // 3):].mean(1)
    f["salto_final"] = np.abs(Pv.max(1) - conf[:, -1])
    # vistas
    dec_v = np.stack([leit[v][1].argmax(1) for v in VISTAS], 1)
    pv = np.stack([leit[v][1][np.arange(n), yhat] for v in VISTAS], 1)
    f["vistas_mudam"] = (dec_v[:, 1:] != yhat[:, None]).mean(1)
    f["p_vistas_std"] = pv.std(1)
    T = np.stack([probes.proba(leit[v][0])[np.arange(n), :, yhat] for v in VISTAS], 0)
    f["trajetoria_vistas"] = T.std(0).mean(1)
    # vizinhos no espaço do último bloco
    U = _unit(A[-1]); S = U @ banco_U.T
    if excluir_si: np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, 1)[:, :k]
    f["vizinhos_discordam"] = (banco_dec[idx] != yhat[:, None]).mean(1)
    f["dist_vizinhos"] = (1 - np.take_along_axis(S, idx, 1)).mean(1) / dist_ref
    f["choquet_fna"] = choquet_fna(conf)
    f["prox_p"] = 1 - Pv.max(1)                             # incerteza do softmax (referência)
    return pd.DataFrame(f), pp, conf


def _unit(a):
    a = a - a.mean(1, keepdims=True)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)


# ----------------------------------------------------------------------------- métricas
def aurc(erro, score):
    """Área sob a curva risco × cobertura (menor = melhor), em ×10^3 como no FD-Shifts."""
    o = np.argsort(score); e = erro[o]
    return 1000 * np.mean(np.cumsum(e) / np.arange(1, len(e) + 1))


def alem_de_p(erro, score, pmax, nb=5):
    q = pd.qcut(pd.Series(pmax).rank(method="first"), nb, labels=False).values; s = n = 0
    for b in range(nb):
        m = q == b
        if 0 < erro[m].sum() < m.sum():
            s += roc_auc_score(erro[m], score[m]) * m.sum(); n += m.sum()
    return s / n if n else np.nan


def metricas(erro, score, pmax, n_boot=200, seed=0):
    base = {"AUROC": roc_auc_score(erro, score), "AUPR-erro": average_precision_score(erro, score),
            "AURC (x1000, menor melhor)": aurc(erro, score), "AUROC além de P": alem_de_p(erro, score, pmax)}
    r = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = r.integers(0, len(erro), len(erro))
        if 0 < erro[i].sum() < len(i): bs.append(roc_auc_score(erro[i], score[i]))
    base["IC95 AUROC"] = f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"
    return base


TOP_K = 10


def ordenar(Pr, top=TOP_K):
    """Probabilidades em ordem decrescente (por probe, se 3D), só as `top` maiores. Como Chen analisa: a 1ª, a 2ª maior, ...
    Sem isso a regressão logística não consegue usar 'a maior probabilidade', que depende da classe prevista.
    No CIFAR-10 top=10 = todas as classes (idêntico a Chen); com 100/200 classes, guarda as 10 maiores (desvio documentado)."""
    return -np.sort(-Pr, axis=-1)[..., :top]


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.15, random_state=0)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


# ----------------------------------------------------------------------------- um cenário completo (limpo ou ruidoso)
CHEN = {  # Tabela 2 de Chen et al. 2019 (AUC). in = TEST; ood = TEST + CIFAR-100
    ("limpo", "in"): {"Softmax": 0.91, "Blackbox LR": 0.91, "Blackbox GBM": 0.91, "Probes LR (Chen)": 0.88, "Probes GBM (Chen)": 0.91},
    ("ruidoso", "in"): {"Softmax": 0.74, "Blackbox LR": 0.79, "Blackbox GBM": 0.80, "Probes LR (Chen)": 0.87, "Probes GBM (Chen)": 0.88},
    ("limpo", "ood"): {"Softmax": 0.89, "Blackbox LR": 0.89, "Blackbox GBM": 0.89, "Probes LR (Chen)": 0.85, "Probes GBM (Chen)": 0.89},
    ("ruidoso", "ood"): {"Softmax": 0.72, "Blackbox LR": 0.76, "Blackbox GBM": 0.77, "Probes LR (Chen)": 0.83, "Probes GBM (Chen)": 0.84},
}


def cenario(nome, frac_ruido, Xtr, ytr, Xte, yte, Xood, device, epocas, k_fuzzy=4, caminho_modelo=None, log=print,
            modelo="chen", passos_chen=90_000, dataset="cifar10", n_classes=10):
    """modelo='chen': ResNet-32 pré-ativação com a receita do exemplo do TensorFlow usado por Chen (17 pontos de leitura).
    modelo='nosso': a ResNet-32 pós-ativação com OneCycle da versão anterior (16 pontos)."""
    import os, regras_fuzzy as rf
    ib, im, idv = dividir(ytr)
    y_ruid = ruido(ytr, frac_ruido, n_classes) if frac_ruido > 0 else ytr.copy()   # ruído em TRAIN-BASE, TRAIN-META e DEV
    model = ResNetChen(n_classes=n_classes) if modelo == "chen" else ResNetCifar(n_classes=n_classes)
    if caminho_modelo and os.path.exists(caminho_modelo):
        model.load_state_dict(torch.load(caminho_modelo, map_location=device)); model.to(device); log(f"[{nome}] modelo carregado")
    else:
        log(f"[{nome}] treinando ResNet-32 ({modelo}) em TRAIN-BASE ({len(ib)} imagens, {frac_ruido:.0%} de rótulos trocados)")
        if modelo == "chen":
            treinar_chen(model, Xtr[ib], y_ruid[ib], Xtr[idv], y_ruid[idv], device, passos_chen, log=log)
        else:
            treinar(model, Xtr[ib], y_ruid[ib], Xtr[idv], y_ruid[idv], epocas, device, log=log)
        if caminho_modelo: torch.save(model.state_dict(), caminho_modelo)
    acc = acuracia(model, Xte, yte, device); log(f"[{dataset}/{nome}] acurácia no TEST (limpo): {acc:.3f}" + (f"  (Chen: {'0.904' if frac_ruido == 0 else '0.774'})" if dataset == "cifar10" else ""))

    t0 = time.time()
    L_meta, L_test = ler_tudo(model, Xtr[im], device), ler_tudo(model, Xte, device)
    L_ood = ler_tudo(model, Xood, device) if Xood is not None else None
    A_meta, P_meta = L_meta["original"]; yhat_meta = P_meta.argmax(1)
    y_meta = y_ruid[im]                                                    # como em Chen: rótulos de TRAIN-META (ruidosos no cenário ruidoso)
    erro_meta = (yhat_meta != y_meta).astype(int)

    pr_chen = Probes().fit(A_meta, y_meta, n_classes)                      # Chen: probes preveem a classe verdadeira
    pr_nos = Probes().fit(A_meta, yhat_meta, n_classes)                    # nosso: probes imitam a decisão da rede
    U = _unit(A_meta[-1]); S = U @ U.T; np.fill_diagonal(S, -np.inf)
    dist_ref = float(np.median((1 - np.sort(S, 1)[:, -15:]).mean(1))) + EPS
    F_meta, _, _ = features_nossas(L_meta, pr_nos, U, yhat_meta, dist_ref, excluir_si=True)
    log(f"[{nome}] leitura da rede e probes: {time.time() - t0:.0f}s")

    def pacote(L, y_verdade, ood=False):
        A, P = L["original"]; F, _, _ = features_nossas(L, pr_nos, U, yhat_meta, dist_ref)
        erro = np.ones(len(P), int) if ood else (P.argmax(1) != y_verdade).astype(int)
        return {"P": P, "Pord": ordenar(P), "chen": ordenar(pr_chen.proba(A)).reshape(len(P), -1), "F": F, "erro": erro}

    tr = {"P": P_meta, "Pord": ordenar(P_meta), "chen": ordenar(pr_chen.proba(A_meta)).reshape(len(P_meta), -1), "F": F_meta, "erro": erro_meta}
    te = pacote(L_test, yte)
    conjuntos = {"in": te}
    if L_ood is not None:
        oo = pacote(L_ood, None, ood=True)
        conjuntos["ood"] = {k: (pd.concat([te[k], oo[k]], ignore_index=True) if k == "F" else np.concatenate([te[k], oo[k]])) for k in te}

    tab_meta = F_meta.assign(erro=erro_meta)
    feats = rf.selecionar(tab_meta, NOSSAS, k_fuzzy)
    feats_wm = rf.selecionar(tab_meta, NOSSAS, 3)                         # Wang-Mendel: 3 features (3^3 = 27 regras no máximo)
    modelos = {
        "Blackbox LR": (lr, "Pord", None), "Blackbox GBM": (gbm, "Pord", None),
        "Probes LR (Chen)": (lr, "chen", None), "Probes GBM (Chen)": (gbm, "chen", None),
        "Nosso: features + LR": (lr, "F", NOSSAS), "Nosso: features + GBM": (gbm, "F", NOSSAS),
        f"Nosso: fuzzy aditivo ({k_fuzzy} feat.)": ("adi", "F", feats), f"Nosso: árvore fuzzy ({k_fuzzy} feat.)": ("arv", "F", feats),
        "Nosso: fuzzy aditivo (todas)": ("adi", "F", NOSSAS),
        "Nosso: Wang-Mendel (3 feat.)": ("wm", "F", feats_wm),
    }
    ajustados = {}
    for m, (fab, chave, cols) in modelos.items():
        Xm = tr[chave] if cols is None else tr[chave][cols].values
        if fab == "adi": ajustados[m] = rf.FuzzyAditivo().fit(Xm, tr["erro"], cols)
        elif fab == "arv": ajustados[m] = rf.ArvoreFuzzy(2).fit(Xm, tr["erro"], cols)
        elif fab == "wm": ajustados[m] = rf.WangMendel().fit(Xm, tr["erro"], cols)
        else: ajustados[m] = fab().fit(Xm, tr["erro"])

    linhas = []
    for dom, d in conjuntos.items():
        pmax = d["P"].max(1)
        scores = {"Softmax": 1 - pmax}
        for m, (fab, chave, cols) in modelos.items():
            Xm = d[chave] if cols is None else d[chave][cols].values
            scores[m] = ajustados[m].predict_proba(Xm)[:, 1]
        for m, s in scores.items():
            linhas.append({"dataset": dataset, "cenário": nome, "conjunto": "TEST" if dom == "in" else "TEST + OOD", "método": m,
                           "AUROC Chen 2019 (publicado)": CHEN[(nome, dom)].get(m, np.nan) if dataset == "cifar10" else np.nan,
                           "uso": "interno (fora do artigo)" if "Wang-Mendel" in m else "artigo",
                           **metricas(d["erro"], s, pmax)})
    res = pd.DataFrame(linhas)
    adi = ajustados[f"Nosso: fuzzy aditivo ({k_fuzzy} feat.)"].regras()
    arv = ajustados[f"Nosso: árvore fuzzy ({k_fuzzy} feat.)"].regras()
    diag = rf.diagnostico(te["F"].assign(erro=te["erro"]), NOSSAS)
    wm = ajustados["Nosso: Wang-Mendel (3 feat.)"].regras()
    return {"res": res, "acc": acc, "feats": feats, "feats_wm": feats_wm, "regras_aditivo": adi, "regras_arvore": arv, "regras_wm": wm,
            "erros_test": int(te["erro"].sum()), "diag": diag}


# ----------------------------------------------------------------------------- datasets
DATASETS = {   # classes, lado da imagem, conjunto fora da distribuição (OOD) usado em TEST + OOD
    "cifar10": {"classes": 10, "lado": 32, "ood": "cifar100"},
    "cifar100": {"classes": 100, "lado": 32, "ood": "cifar10"},
    "tinyimagenet": {"classes": 200, "lado": 64, "ood": None},
}
URL_TINY = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _cifar(nome, pasta, treino):
    import torchvision
    ds = (torchvision.datasets.CIFAR10 if nome == "cifar10" else torchvision.datasets.CIFAR100)(pasta, train=treino, download=True)
    return ds.data, np.array(ds.targets)


def baixar_tiny(pasta):
    """Baixa e descompacta o Tiny-ImageNet (~240 MB) se ainda não existir. Devolve a pasta tiny-imagenet-200."""
    import os, urllib.request, zipfile
    raiz = os.path.join(pasta, "tiny-imagenet-200")
    if not os.path.isdir(raiz):
        os.makedirs(pasta, exist_ok=True); z = os.path.join(pasta, "tiny-imagenet-200.zip")
        if not os.path.exists(z):
            print("baixando Tiny-ImageNet..."); urllib.request.urlretrieve(URL_TINY, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(pasta)
    return raiz


def carregar_tiny(raiz):
    """Treino: 500 imagens × 200 classes. TEST = o conjunto 'val' oficial (10k, com rótulos), pois o 'test' oficial não tem rótulos.
    Classes na ordem de wnids.txt."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    wnids = [l.strip() for l in open(os.path.join(raiz, "wnids.txt")) if l.strip()]
    idx = {w: i for i, w in enumerate(wnids)}
    tr = [(os.path.join(raiz, "train", w, "images", f), idx[w]) for w in wnids
          for f in sorted(os.listdir(os.path.join(raiz, "train", w, "images")))]
    va = [(os.path.join(raiz, "val", "images", l.split("\t")[0]), idx[l.split("\t")[1]])
          for l in open(os.path.join(raiz, "val", "val_annotations.txt")) if l.strip()]
    ler = lambda c: np.asarray(Image.open(c).convert("RGB"), dtype=np.uint8)
    with ThreadPoolExecutor(8) as ex:
        Xtr = np.stack(list(ex.map(ler, [c for c, _ in tr]))); Xte = np.stack(list(ex.map(ler, [c for c, _ in va])))
    return Xtr, np.array([y for _, y in tr]), Xte, np.array([y for _, y in va])


def carregar(nome, pasta, tiny_dir=None, com_ood=True):
    """Devolve Xtr, ytr, Xte, yte (uint8 N×H×W×3), Xood (ou None) e o nº de classes."""
    info = DATASETS[nome]
    if nome in ("cifar10", "cifar100"):
        Xtr, ytr = _cifar(nome, pasta, True); Xte, yte = _cifar(nome, pasta, False)
    else:
        Xtr, ytr, Xte, yte = carregar_tiny(tiny_dir or baixar_tiny(pasta))
    Xood = _cifar(info["ood"], pasta, False)[0] if (com_ood and info["ood"]) else None
    return Xtr, ytr, Xte, yte, Xood, info["classes"]


# ----------------------------------------------------------------------------- tudo de uma vez (usado pelo notebook e pelo rodar.py)
ORDEM = ['Softmax', 'Blackbox LR', 'Blackbox GBM', 'Probes LR (Chen)', 'Probes GBM (Chen)', 'Nosso: features + LR',
         'Nosso: features + GBM', 'Nosso: fuzzy aditivo ({k} feat.)', 'Nosso: árvore fuzzy ({k} feat.)', 'Nosso: fuzzy aditivo (todas)']


def rodar_tudo(datasets=("cifar10", "cifar100", "tinyimagenet"), cenarios=(("limpo", 0.0), ("ruidoso", 0.3)), modelo="chen",
               passos=90_000, epocas=60, k_fuzzy=4, com_ood=True, dados="./dados", tiny_dir=None, pasta_modelos=".", device=None, log=print):
    """Para cada dataset e cenário: treina (ou carrega) a ResNet, lê a rede, treina os meta-modelos no TRAIN-META e avalia no TEST.
    Devolve (saidas, res, protocolo): saidas[(dataset, cenário)] = dicionário de cenario(); res = todas as métricas; protocolo = o que foi usado."""
    import gc, os
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    saidas, protocolo = {}, []
    for ds in datasets:
        Xtr, ytr, Xte, yte, Xood, K = carregar(ds, dados, tiny_dir, com_ood)
        ib, im, idv = dividir(ytr)
        log(f"== {ds}: treino {Xtr.shape}, TEST {Xte.shape}, {K} classes, OOD {None if Xood is None else Xood.shape}")
        for nome, frac in cenarios:
            s = cenario(nome, frac, Xtr, ytr, Xte, yte, Xood, device, epocas, k_fuzzy=k_fuzzy,
                        caminho_modelo=os.path.join(pasta_modelos, f"resnet32_{modelo}_{ds}_{nome}.pt"), log=log,
                        modelo=modelo, passos_chen=passos, dataset=ds, n_classes=K)
            saidas[(ds, nome)] = s
            protocolo.append({"dataset": ds, "cenário": nome, "classes": K, "lado (px)": Xtr.shape[1],
                              "TRAIN-BASE": len(ib), "TRAIN-META": len(im), "DEV": len(idv), "TEST": len(Xte),
                              "OOD": DATASETS[ds]["ood"] if com_ood else None, "rótulos trocados": frac,
                              "modelo": modelo, "passos/épocas": passos if modelo == "chen" else epocas,
                              "acurácia TEST": s["acc"],
                              "acurácia de Chen": {"limpo": 0.904, "ruidoso": 0.774}.get(nome) if ds == "cifar10" else None,
                              "erros TEST": s["erros_test"], "features fuzzy": ", ".join(s["feats"])})
            log(f"{ds}/{nome}: {s['erros_test']} erros no TEST | features fuzzy: {s['feats']}")
        del Xtr, Xte, Xood; gc.collect()
    res = pd.concat([s["res"] for s in saidas.values()], ignore_index=True)
    return saidas, res, pd.DataFrame(protocolo)


def salvar(saidas, res, protocolo, arquivo="classicos_resultados.xlsx"):
    """Abas principais = o que vai para o artigo; Wang-Mendel só nas abas interno_*."""
    art, wm = res[res.uso == "artigo"], res[res.uso != "artigo"]
    with pd.ExcelWriter(arquivo) as xw:
        protocolo.round(4).to_excel(xw, sheet_name="protocolo_usado", index=False)
        art.round(4).to_excel(xw, sheet_name="comparacao", index=False)
        for (ds, nome), s in saidas.items():
            s["regras_aditivo"].to_excel(xw, sheet_name=f"regras_{ds}_{nome}"[:31], index=False)
            s["diag"].round(4).to_excel(xw, sheet_name=f"diag_{ds}_{nome}"[:31], index=False)
        wm.round(4).to_excel(xw, sheet_name="interno_wang_mendel", index=False)
        for (ds, nome), s in saidas.items():
            s["regras_wm"].to_excel(xw, sheet_name=f"interno_wm_{ds}_{nome}"[:31], index=False)
    return arquivo
