"""
fuzzy_outros.py — os métodos FUZZY de outros autores (família 8 da comparação), adaptados à detecção de erros
da rede. Nenhum deles foi proposto exatamente para esta tarefa; cada adaptação está descrita abaixo e em
experimentos/PROTOCOLO.md (seção E). Todos produzem um escore: MAIOR = MAIS CHANCE DE A REDE TER ERRADO.

  barna2026        Barna, Petcut e Petcut-Lasc, Applied Sciences 16(21) 10568, 2026.
                   Abstenção fuzzy-possibilística sobre a saída: similaridade de Goguen entre as duas classes mais
                   prováveis e baixa confiança, agregadas pelo máximo (t-conorma). Versão "só softmax" (o artigo usa
                   também a distância da câmera, que não existe aqui). Sem treino.
                   FÓRMULA RECONSTRUÍDA da descrição do artigo; conferir com o PDF.

  Aghaeipoor2023   Aghaeipoor, Sabokrou e Fernández, IEEE TFS 2023. Sistemas de regras fuzzy que explicam a rede a
                   partir de valores de atribuição. Adaptação: atribuição gradiente×entrada na penúltima camada,
                   resumida em 6 números (a penúltima camada tem centenas de dimensões), e regras Wang–Mendel
                   aprendidas para prever o erro.

  ENNreg           Denoeux, IEEE TFS 2023 ("ENNreg", números fuzzy aleatórios gaussianos). Adaptação: protótipos
                   evidenciais prevêem a confiança na classe verdadeira (TCP, como o ConfidNet) com média, variância
                   e precisão; combinação pela regra produto-interseção normalizada; treino por verossimilhança
                   gaussiana (simplificação da perda do artigo). Escore = 1 − limite inferior (90%) da TCP prevista.

  Msaddi2026       Msaddi e Kumbasar, IEEE TETCI 2026 ("Alliance": fuzzy tipo 2 intervalar + predição conforme).
                   Sistema TSK tipo 2 intervalar (gaussianas com FOU, redução de tipo Karnik–Mendel), treinado por
                   regressão quantílica (perda pinball nos quantis 5% e 95% + log-cosh), calibrado por CQR numa parte
                   separada do treino. Alvo: TCP. Escore = 1 − limite inferior calibrado.

Para a comparação ser justa, ENNreg e Msaddi2026 recebem as MESMAS características que o nosso fuzzy aditivo
(P + 3 escolhidas no mesmo fold). Barna2026 usa só a saída; Aghaeipoor2023 usa só as atribuições.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

EPS = 1e-9


# ============================================================================ Barna et al. 2026
def barna2026(probs, dec):
    """probs (n, K); dec = classe decidida. Similaridade de Goguen entre as duas mais prováveis (p2/p1, em [0,1])
    e baixa confiança (1 − p1), agregadas pelo máximo: possibilidade de erro."""
    r = np.arange(len(probs))
    p1 = probs[r, dec]
    outras = probs.copy(); outras[r, dec] = -1
    p2 = outras.max(1)
    sim = np.clip(p2 / np.maximum(p1, EPS), 0, 1)
    return np.maximum(sim, 1 - p1)


# ============================================================================ Aghaeipoor et al. 2023
NOMES_ATRIB = ["atrib_total", "atrib_positiva", "atrib_negativa", "atrib_concentracao", "atrib_entropia", "atrib_frac_positiva"]


def resumo_atribuicoes(A):
    """A (n, D) = atribuição gradiente×entrada de cada unidade da penúltima camada para o logit decidido."""
    a = np.asarray(A, float); ab = np.abs(a) + EPS; pab = ab / ab.sum(1, keepdims=True)
    top = np.sort(ab, 1)[:, ::-1][:, :5].sum(1) / ab.sum(1)
    ent = -(pab * np.log(pab)).sum(1) / np.log(a.shape[1])
    return np.c_[a.sum(1), np.clip(a, 0, None).sum(1), np.clip(a, None, 0).sum(1), top, ent, (a > 0).mean(1)]


def _escolher_por_auroc(X, y, k=3, max_corr=0.85):
    from scipy.stats import spearmanr
    forca = [abs(roc_auc_score(y, X[:, j]) - .5) if np.std(X[:, j]) > 0 else 0 for j in range(X.shape[1])]
    esc = []
    for j in np.argsort(forca)[::-1]:
        if all(abs(spearmanr(X[:, j], X[:, i]).correlation) < max_corr for i in esc):
            esc.append(int(j))
        if len(esc) == k:
            break
    return esc


class Aghaeipoor2023:
    def fit(self, Xa, y):
        import fuzzy as fz
        self.cols = _escolher_por_auroc(Xa, y)
        self.wm = fz.WangMendel().fit(Xa[:, self.cols], y, [NOMES_ATRIB[j] for j in self.cols])
        return self

    def score(self, Xa):
        return self.wm.predict_proba(Xa[:, self.cols])[:, 1]


# ============================================================================ utilidades de treino
def _treinar(modulo, perda, X, y, epocas=400, lr=0.02, wd=1e-4, semente=0):
    torch.manual_seed(semente)
    Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(modulo.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epocas):
        opt.zero_grad(); l = perda(modulo(Xt), yt); l.backward(); opt.step()
    return modulo


def _iniciar_centros(X, K, semente):
    r = np.random.default_rng(semente)
    return torch.as_tensor(X[r.choice(len(X), K, replace=len(X) < K)], dtype=torch.float32)


# ============================================================================ ENNreg (Denoeux 2023)
class _ENNreg(nn.Module):
    def __init__(self, C):
        super().__init__()
        K, M = C.shape
        self.p = nn.Parameter(C.clone())                       # protótipos
        self.g = nn.Parameter(torch.full((K,), 0.5))           # escala da distância (γ)
        self.a = nn.Parameter(torch.zeros(K))                  # importância do protótipo (α, via sigmoide)
        self.beta = nn.Parameter(torch.zeros(K, M)); self.b0 = nn.Parameter(torch.full((K,), 0.5))
        self.ls = nn.Parameter(torch.full((K,), -1.5))         # log σ_k
        self.lh = nn.Parameter(torch.zeros(K))                 # log h_k (precisão)

    def forward(self, x):
        d2 = ((x[:, None, :] - self.p[None]) ** 2).sum(2)
        s = torch.sigmoid(self.a) * torch.exp(-(self.g ** 2) * d2)               # ativação de cada protótipo
        mu_k = x @ self.beta.T + self.b0
        hk = torch.exp(self.lh); sk2 = torch.exp(2 * self.ls)
        h = (s * hk).sum(1) + EPS                                                  # produto-interseção normalizado
        mu = (s * hk * mu_k).sum(1) / h
        var = ((s * hk) ** 2 * sk2).sum(1) / h ** 2
        return mu, var, h


class ENNreg:
    def __init__(self, K=10, z=1.645, semente=0):
        self.K, self.z, self.semente = K, z, semente

    def fit(self, X, tcp):
        self.sc = StandardScaler().fit(X); Xs = self.sc.transform(X)
        self.m = _ENNreg(_iniciar_centros(Xs, self.K, self.semente))

        def perda(saida, y):
            mu, var, h = saida
            v = var + 1 / h + 1e-4                                                 # aleatória + epistêmica
            return (0.5 * torch.log(v) + 0.5 * (y - mu) ** 2 / v).mean()
        _treinar(self.m, perda, Xs, tcp, semente=self.semente)
        return self

    @torch.no_grad()
    def score(self, X):
        mu, var, h = self.m(torch.as_tensor(self.sc.transform(X), dtype=torch.float32))
        inf = mu - self.z * torch.sqrt(var + 1 / h)
        return (1 - inf).numpy()


# ============================================================================ Msaddi e Kumbasar 2026
class _IT2TSK(nn.Module):
    """TSK tipo 2 intervalar: antecedentes gaussianos com FOU (desvio inferior ≤ superior, altura ≤ 1),
    consequentes lineares, redução de tipo Karnik–Mendel exata (testa todos os pontos de troca)."""

    def __init__(self, C):
        super().__init__()
        P, M = C.shape
        self.M = M
        self.c = nn.Parameter(C.clone())
        self.sig = nn.Parameter(torch.zeros(P, M))             # σ superior = softplus(.) + 0,1
        self.r = nn.Parameter(torch.full((P, M), 1.0))         # σ inferior = σ superior · sigmoide(r)
        self.hh = nn.Parameter(torch.full((P, M), 2.0))        # altura da inferior = sigmoide(hh)
        self.a = nn.Parameter(torch.zeros(P, M)); self.a0 = nn.Parameter(torch.full((P,), 0.5))

    def forward(self, x):
        su = nn.functional.softplus(self.sig) + 0.1
        sl = su * torch.sigmoid(self.r)
        d2 = (x[:, None, :] - self.c[None]) ** 2
        esc = 2 * self.M
        fu = torch.exp(-(d2 / (esc * su ** 2)).sum(2))
        fl = torch.exp((torch.log(torch.sigmoid(self.hh)) - d2 / (esc * sl ** 2)).sum(2))
        y = x @ self.a.T + self.a0                                                  # (n, P)
        o = torch.argsort(y, 1); y = torch.gather(y, 1, o); fu = torch.gather(fu, 1, o); fl = torch.gather(fl, 1, o)
        cu_y, cl_y = torch.cumsum(fu * y, 1), torch.cumsum(fl * y, 1)
        cu, cl = torch.cumsum(fu, 1), torch.cumsum(fl, 1)
        z = torch.zeros_like(y[:, :1])
        cu_y, cl_y, cu, cl = [torch.cat([z, t], 1) for t in (cu_y, cl_y, cu, cl)]   # somas até L (L = 0..P)
        tot_uy, tot_ly, tot_u, tot_l = cu_y[:, -1:], cl_y[:, -1:], cu[:, -1:], cl[:, -1:]
        y_inf = ((cu_y + (tot_ly - cl_y)) / (cu + (tot_l - cl) + EPS)).min(1).values     # superior até L, inferior depois
        y_sup = ((cl_y + (tot_uy - cu_y)) / (cl + (tot_u - cu) + EPS)).max(1).values     # inferior até R, superior depois
        return y_inf, y_sup


class Msaddi2026:
    def __init__(self, P=8, cobertura=0.9, frac_cal=0.25, semente=0):
        self.P, self.d, self.frac_cal, self.semente = P, cobertura, frac_cal, semente

    def fit(self, X, tcp):
        r = np.random.default_rng(self.semente); idx = r.permutation(len(X)); q = max(5, int(self.frac_cal * len(X)))
        cal, tr = idx[:q], idx[q:]
        self.sc = StandardScaler().fit(X[tr]); Xs = self.sc.transform(X)
        self.m = _IT2TSK(_iniciar_centros(Xs[tr], self.P, self.semente))
        ti, ts = (1 - self.d) / 2, (1 + self.d) / 2

        def pinball(e, t):
            return torch.maximum(t * e, (t - 1) * e)

        def perda(saida, y):
            yi, ys = saida
            return (pinball(y - yi, ti) + pinball(y - ys, ts) + torch.log(torch.cosh(y - (yi + ys) / 2))).mean()
        _treinar(self.m, perda, Xs[tr], tcp[tr], semente=self.semente)
        with torch.no_grad():                                                       # calibração conforme (CQR)
            yi, ys = self.m(torch.as_tensor(Xs[cal], dtype=torch.float32))
        E = np.maximum(yi.numpy() - tcp[cal], tcp[cal] - ys.numpy())
        s = min(len(E), int(np.ceil(self.d * (len(E) + 1))))
        self.alfa = float(np.sort(E)[s - 1])
        return self

    @torch.no_grad()
    def score(self, X):
        yi, _ = self.m(torch.as_tensor(self.sc.transform(X), dtype=torch.float32))
        return 1 - (yi.numpy() - self.alfa)


# ============================================================================ atribuições (para Aghaeipoor 2023)
def atribuicoes(rede, pen, dec, disp="cpu", lote=2048):
    """Gradiente×entrada do logit da classe decidida em relação à penúltima camada (a cabeça da rede é diferenciável)."""
    cab = rede.cabeca; cab.eval(); out = []
    for i in range(0, len(pen), lote):
        x = torch.as_tensor(pen[i:i + lote], device=disp).float().requires_grad_(True)
        z = cab(x).float()
        d = torch.as_tensor(dec[i:i + lote], device=disp).long()
        alvo = z[:, 0] * (2 * d - 1).float() if z.shape[1] == 1 else z.gather(1, d[:, None])[:, 0]
        g, = torch.autograd.grad(alvo.sum(), x)
        out.append((g * x).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)
