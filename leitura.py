"""
leitura.py — lê a rede TREINADA e CONGELADA por dentro e calcula, para cada imagem de teste:

NOSSAS FEATURES (comportamento interno da rede, nada medido nos pixels)
  1) Como chegou na resposta? (probes que imitam a DECISÃO da rede em cada bloco)
     trocas_opiniao   quantas vezes a opinião das camadas troca entre concordar e discordar da resposta final
     prof_decisao     a partir de que fração da rede a opinião não muda mais (Baldock et al. 2021)
     entropia_probes  indecisão média das camadas (entropia normalizada, 0 a 1)
     conf_fim         o quanto o último terço da rede já concorda com a resposta final
     salto_final      a cabeça da rede discorda do último bloco
  2) A resposta é estável? (vistas: espelhos/deslocamentos — o objeto é o mesmo)
     vistas_mudam, p_vistas_std, trajetoria_vistas
  3) Já viu algo parecido? (vizinhos do treino na penúltima camada)
     vizinhos_discordam, dist_vizinhos
  p  a resposta da rede: binário = P(maligna); multiclasse = probabilidade da classe escolhida

MÉTODOS DE OUTROS ARTIGOS (calculados aqui; os que aprendem são treinados em comparacao.py)
  msp, entropia_softmax          Hendrycks & Gimpel 2017
  prox_p                         distância ao limiar (binário)
  mc_dropout_std, mc_entropia    Gal & Ghahramani 2016 (Dropout da cabeça ligado, T repetições)
  mahalanobis                    Lee et al. 2018 (penúltima camada, covariância comum)
  font_dc, font_lu               Font et al. 2025 (vizinhos em cada camada, Bray-Curtis)
  extras["chen"]                 Chen et al. 2019 (probes treinados com o rótulo, top-3 por camada)
  extras["penultima"]            para Gorbett & Blanchard 2022 e ConfidNet (Corbière 2019)
"""
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from dados import carregador
from redes import probabilidades

EPS = 1e-9
NOSSAS = ["trocas_opiniao", "prof_decisao", "entropia_probes", "conf_fim", "salto_final",
          "vistas_mudam", "p_vistas_std", "trajetoria_vistas", "vizinhos_discordam", "dist_vizinhos", "p"]
CLASSICAS = ["msp", "entropia_softmax", "prox_p", "mc_dropout_std", "mc_entropia", "mahalanobis", "font_dc", "font_lu"]


def _entropia(P):
    P = np.clip(P, EPS, 1)
    return -(P * np.log(P)).sum(1) / np.log(P.shape[1])


def _prob_da_classe(pr, X, classe):
    """Probabilidade que o probe dá à `classe` de cada linha (0 se o probe nunca viu a classe)."""
    proba = pr.predict_proba(X)
    col = {c: j for j, c in enumerate(pr.classes_)}
    idx = np.array([col.get(c, -1) for c in classe])
    out = np.where(idx >= 0, proba[np.arange(len(X)), np.maximum(idx, 0)], 0.0)
    return out, proba


class LeitorRede:
    def __init__(self, rede, n_classes, limiar=None, vistas=None, disp="cpu", lote=128):
        self.rede, self.K, self.limiar, self.disp, self.lote = rede, n_classes, limiar, disp, lote
        self.vistas = vistas or {"original": lambda x: x}
        self.B = len(rede.blocos)

    # ------------------------------------------------------------------ leitura crua
    @torch.no_grad()
    def _ler(self, ds, vista=None):
        self.rede.eval()
        acts, pen, probs = [[] for _ in range(self.B)], [], []
        cap = [None] * self.B
        reduz = lambda o: (o.float().mean((2, 3)) if o.dim() == 4 else o.float()).cpu().numpy()
        hs = [b.register_forward_hook(lambda m, i, o, k=k: cap.__setitem__(k, reduz(o))) for k, b in enumerate(self.rede.blocos)]
        hp = self.rede.cabeca.register_forward_pre_hook(lambda m, i: pen.append(i[0].float().cpu().numpy()))
        try:
            for x, _, _ in carregador(ds, self.lote):
                x = x.to(self.disp, non_blocking=True)
                probs.append(probabilidades(self.rede(vista(x) if vista else x).float()).cpu().numpy())
                for k in range(self.B):
                    acts[k].append(cap[k])
        finally:
            for h in hs + [hp]:
                h.remove()
        return [np.concatenate(a).astype(np.float32) for a in acts], np.concatenate(pen).astype(np.float32), np.concatenate(probs)

    def _decisao(self, probs):
        return (probs[:, 1] >= self.limiar).astype(int) if self.K == 2 and self.limiar is not None else probs.argmax(1)

    @staticmethod
    def _probe(X, y):
        passos = [StandardScaler()]
        n_comp = min(C.PCA_PROBES or 10**9, len(X) - 1)
        if C.PCA_PROBES and X.shape[1] > n_comp:
            passos.append(PCA(n_comp, random_state=0))
        return make_pipeline(*passos, LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")).fit(X, y)

    # ------------------------------------------------------------------ referência (treino)
    def fit(self, ds_ref, y_ref, log=print):
        A, pen, probs = self._ler(ds_ref)
        y_ref = np.asarray(y_ref).astype(int)
        dec = self._decisao(probs)
        alvo = dec if len(np.unique(dec)) > 1 else y_ref
        log(f"  probes: {self.B} blocos × 2 (imitam a decisão / Chen: imitam o rótulo)")
        self.probes = [self._probe(a, alvo) for a in A]
        self.probes_chen = [self._probe(a, y_ref) for a in A]
        self.banco, self.banco_dec = self._unit(pen), dec                       # vizinhos
        d = self._knn(self.banco, excluir_si=True)[1]
        self.dist_ref = float(np.median(d)) + EPS
        classes = np.unique(y_ref)                                              # Mahalanobis
        self.centros = np.stack([pen[y_ref == c].mean(0) for c in classes])
        self.precisao = LedoitWolf().fit(pen - self.centros[np.searchsorted(classes, y_ref)]).precision_.astype(np.float32)
        self.A_ref, self.y_ref = A, y_ref                                       # Font et al.
        return self

    @staticmethod
    def _unit(a):
        a = a - a.mean(1, keepdims=True)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)

    def _knn(self, U, excluir_si=False, k=None, bloco=2000):
        k = k or C.K_VIZINHOS; idx, dist = [], []
        for i in range(0, len(U), bloco):
            S = U[i:i + bloco] @ self.banco.T
            if excluir_si:
                S[np.arange(len(S)), np.arange(i, i + len(S))] = -np.inf
            j = np.argpartition(-S, k, 1)[:, :k]
            idx.append(j); dist.append((1 - np.take_along_axis(S, j, 1)).mean(1))
        return np.concatenate(idx), np.concatenate(dist)

    def _mahalanobis(self, pen, bloco=2000):
        out = []
        for i in range(0, len(pen), bloco):
            a = pen[i:i + bloco]
            out.append(np.min([np.einsum("ij,jk,ik->i", a - m, self.precisao, a - m) for m in self.centros], axis=0))
        return np.sqrt(np.maximum(np.concatenate(out), 0))

    @torch.no_grad()
    def _mc_dropout(self, pen, semente=0):
        cab = self.rede.cabeca
        x = torch.as_tensor(pen, device=self.disp)
        torch.manual_seed(semente); cab.train()
        try:
            P = np.stack([probabilidades(cab(x).float()).cpu().numpy() for _ in range(C.T_MC)], 0)   # (T, n, C)
        finally:
            cab.eval()
        return P

    def _font(self, A, bloco=1000):
        sub = np.random.default_rng(0).choice(len(self.y_ref), min(C.N_REF_FONT, len(self.y_ref)), replace=False)
        y_ref = self.y_ref[sub]; dom, ent = [], []
        for a, ar in zip(A, [r[sub] for r in self.A_ref]):
            viz = np.concatenate([np.argpartition(cdist(a[i:i + bloco], ar, "braycurtis"), C.K_FONT, 1)[:, :C.K_FONT]
                                  for i in range(0, len(a), bloco)])
            rot = y_ref[viz]
            cont = np.stack([(rot == c).sum(1) for c in np.unique(y_ref)], 1) / C.K_FONT
            dom.append(cont.argmax(1)); ent.append(_entropia(cont) if cont.shape[1] > 1 else np.zeros(len(a)))
        dom = np.stack(dom, 1)
        return (dom[:, 1:] != dom[:, :-1]).sum(1), np.stack(ent, 1).mean(1)

    # ------------------------------------------------------------------ teste
    def transform(self, ds, log=print):
        nomes = list(self.vistas)
        A, pen, probs = self._ler(ds)
        dec = self._decisao(probs)
        n = len(dec); r = np.arange(n)
        conf_rede = probs[r, dec]
        f = {"decisao": dec, "conf_rede": conf_rede}

        # 1) ao longo das camadas
        pp, ents, concorda = [], [], []
        for pr, a in zip(self.probes, A):
            c, proba = _prob_da_classe(pr, a, dec)
            pp.append(c); ents.append(_entropia(proba) if proba.shape[1] > 1 else np.zeros(n))
            concorda.append(pr.classes_[proba.argmax(1)] == dec)
        pp, concorda = np.stack(pp, 1), np.stack(concorda, 1).astype(int)
        f["trocas_opiniao"] = np.abs(np.diff(concorda, axis=1)).sum(1)
        f["prof_decisao"] = np.where(concorda == 0, np.arange(self.B)[None] + 1, 0).max(1) / self.B
        f["entropia_probes"] = np.stack(ents, 1).mean(1)
        f["conf_fim"] = pp[:, -max(1, self.B // 3):].mean(1)
        f["salto_final"] = np.abs(conf_rede - pp[:, -1])

        # 2) entre vistas
        Pv, T = [probs[r, dec]], [pp]
        mudam = []
        for v in nomes[1:]:
            Av, _, pv = self._ler(ds, self.vistas[v])
            Pv.append(pv[r, dec]); mudam.append(self._decisao(pv) != dec)
            T.append(np.stack([_prob_da_classe(pr, a, dec)[0] for pr, a in zip(self.probes, Av)], 1))
            del Av
        f["vistas_mudam"] = np.mean(mudam, 0) if mudam else np.zeros(n)
        f["p_vistas_std"] = np.stack(Pv, 1).std(1)
        f["trajetoria_vistas"] = np.stack(T, 0).std(0).mean(1)

        # 3) vizinhos
        idx, dist = self._knn(self._unit(pen))
        f["vizinhos_discordam"] = (self.banco_dec[idx] != dec[:, None]).mean(1)
        f["dist_vizinhos"] = dist / self.dist_ref
        f["p"] = probs[:, 1] if self.K == 2 else conf_rede

        # clássicos
        f["msp"] = 1 - probs.max(1)
        f["entropia_softmax"] = _entropia(probs)
        if self.K == 2 and self.limiar is not None:
            t, p1 = self.limiar, probs[:, 1]
            f["prox_p"] = np.where(p1 < t, p1 / t, (1 - p1) / (1 - t))
        else:
            f["prox_p"] = f["msp"]
        Pm = self._mc_dropout(pen)
        f["mc_dropout_std"] = Pm[:, r, dec].std(0)
        f["mc_entropia"] = _entropia(Pm.mean(0))
        f["mahalanobis"] = self._mahalanobis(pen)
        log("  Font et al. (vizinhos em cada camada)...")
        f["font_dc"], f["font_lu"] = self._font(A)
        for k in range(self.B):
            f[f"probe_b{k}"] = pp[:, k]

        top = min(self.K, 3) if self.K > 2 else 1
        chen = np.concatenate([np.sort(pr.predict_proba(a), 1)[:, ::-1][:, :top] for pr, a in zip(self.probes_chen, A)], 1)
        extras = {"penultima": pen, "chen": chen.astype(np.float32), "probs": probs.astype(np.float32)}
        return pd.DataFrame(f), extras
