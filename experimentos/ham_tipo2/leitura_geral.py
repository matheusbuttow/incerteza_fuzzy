"""
leitura_geral.py — lê o comportamento interno de QUALQUER rede (EfficientNet, ResNet, ViT, ConvNeXt).

O leitor não sabe o que cada bloco "enxerga". Ele só precisa:
  1. dos pontos de leitura (saída de cada bloco), vindos de redes.py;
  2. de reduzir cada saída a um vetor: média espacial (CNN) ou token de classe (ViT);
  3. de uma probe (regressão logística) por ponto, que diz o que a rede responderia se parasse ali.

Para as features valerem entre redes com números diferentes de blocos, a PROFUNDIDADE é medida de 0 a 1:
  taxa_trocas      trocas de opinião / (nº de blocos − 1)   (a contagem crua depende do nº de blocos)
  prof_decisao     fração da rede a partir da qual a opinião não muda mais
  entropia_probes  indecisão média das camadas
  conf_fim         concordância com a resposta final no último terço da rede
  salto_final      a cabeça discorda do último bloco
  vistas_mudam, p_vistas_std, trajetoria_vistas   estabilidade entre versões espelhadas/giradas
  vizinhos_discordam, dist_vizinhos               familiaridade (vizinhos no último ponto de leitura)
  prox_p           proximidade de P ao limiar (a referência a superar)
Também devolve a trajetória reamostrada em 10 profundidades relativas (traj_r0..traj_r9), igual para todas as redes.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from redes import lotes, preparar

EPS = 1e-9
FEATURES = ["taxa_trocas", "prof_decisao", "entropia_probes", "conf_fim", "salto_final",
            "vistas_mudam", "p_vistas_std", "trajetoria_vistas", "vizinhos_discordam", "dist_vizinhos", "prox_p"]
VISTAS = {"original": lambda x: x, "espelho_h": lambda x: torch.flip(x, [3]),
          "espelho_v": lambda x: torch.flip(x, [2]), "giro_180": lambda x: torch.flip(x, [2, 3])}
N_REL = 10


def reduzir(o):
    """Saída de um bloco → um vetor por imagem."""
    if isinstance(o, (tuple, list)):
        o = o[0]
    if o.dim() == 4:
        return o.float().mean((2, 3))      # CNN: média espacial
    if o.dim() == 3:
        return o[:, 0].float()             # ViT: token de classe
    return o.flatten(1).float()


class LeitorGeral:
    def __init__(self, rede, limiar=0.5, k_vizinhos=15, bs=64):
        self.rede, self.limiar, self.k, self.bs = rede, limiar, k_vizinhos, bs
        self.pontos = list(rede.pontos)
        self.B = len(self.pontos)

    @torch.no_grad()
    def _ler(self, cache, idx, device, vista=lambda x: x):
        self.rede.eval()
        acum, P = [[] for _ in range(self.B)], []
        for xb in lotes(cache, idx, self.bs):
            cap = [None] * self.B
            hs = [b.register_forward_hook(lambda m, i, o, k=k: cap.__setitem__(k, reduzir(o).cpu().numpy()))
                  for k, b in enumerate(self.pontos)]
            try:
                with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                    z = self.rede(vista(preparar(xb, device)))
                P.append(torch.sigmoid(z.float()).cpu().numpy())
            finally:
                for h in hs:
                    h.remove()
            for k in range(self.B):
                acum[k].append(cap[k])
        return [np.concatenate(a).astype(np.float32) for a in acum], np.concatenate(P)

    def fit(self, cache, idx_ref, y_ref, device):
        """Probes e banco de vizinhos a partir de imagens que a rede viu no treino."""
        A, P = self._ler(cache, idx_ref, device)
        dec = (P >= self.limiar).astype(int)
        alvo = dec if len(np.unique(dec)) == 2 else np.asarray(y_ref).astype(int)
        self.probes = [make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
                       .fit(a, alvo) for a in A]
        self.banco, self.banco_dec = self._unit(A[-1]), dec
        self.dist_ref = float(np.median(self._knn_dist(self.banco, excluir_si=True))) + EPS
        return self

    @staticmethod
    def _unit(a):
        a = a - a.mean(1, keepdims=True)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)

    def _knn(self, U, excluir_si=False):
        S = U @ self.banco.T
        if excluir_si:
            np.fill_diagonal(S, -np.inf)
        k = min(self.k, S.shape[1] - (1 if excluir_si else 0))
        return np.argsort(-S, 1)[:, :k], S

    def _knn_dist(self, U, excluir_si=False):
        idx, S = self._knn(U, excluir_si)
        return (1 - np.take_along_axis(S, idx, 1)).mean(1)

    def _trajetoria(self, A):
        return np.stack([pr.predict_proba(a)[:, 1] for pr, a in zip(self.probes, A)], 1)   # (n, B)

    def transform(self, cache, idx, device):
        res = {nome: self._ler(cache, idx, device, v) for nome, v in VISTAS.items()}
        A, P = res["original"]
        pp = self._trajetoria(A)
        B = self.B
        final = (P >= self.limiar).astype(int)
        f = {"p": P}
        op = (pp >= 0.5).astype(int)
        trocas = np.abs(np.diff(op, axis=1)).sum(1)
        f["trocas_opiniao"] = trocas                                   # contagem crua (só para consulta)
        f["taxa_trocas"] = trocas / max(B - 1, 1)
        f["prof_decisao"] = np.where(op != final[:, None], np.arange(B)[None, :] + 1, 0).max(1) / B
        f["entropia_probes"] = -(pp * np.log(pp + EPS) + (1 - pp) * np.log(1 - pp + EPS)).mean(1)
        conf = np.where(final[:, None] == 1, pp, 1 - pp)
        f["conf_fim"] = conf[:, -int(np.ceil(B / 3)):].mean(1)
        f["salto_final"] = np.abs(P - pp[:, -1])
        Ps = np.stack([res[v][1] for v in VISTAS], 1)
        f["vistas_mudam"] = ((Ps[:, 1:] >= self.limiar).astype(int) != final[:, None]).mean(1)
        f["p_vistas_std"] = Ps.std(1)
        T = np.stack([self._trajetoria(res[v][0]) for v in VISTAS], 0)
        f["trajetoria_vistas"] = T.std(0).mean(1)
        idx_v, S = self._knn(self._unit(A[-1]))
        f["vizinhos_discordam"] = (self.banco_dec[idx_v] != final[:, None]).mean(1)
        f["dist_vizinhos"] = (1 - np.take_along_axis(S, idx_v, 1)).mean(1) / self.dist_ref
        f["prox_p"] = 1 - np.abs(P - self.limiar) / max(self.limiar, 1 - self.limiar)
        # trajetória em profundidade relativa (0 = primeiro bloco, 1 = último), comparável entre redes
        x_rel, grade = np.linspace(0, 1, B), np.linspace(0, 1, N_REL)
        conf_rel = np.stack([np.interp(grade, x_rel, c) for c in conf])
        for j in range(N_REL):
            f[f"traj_r{j}"] = conf_rel[:, j]
        return pd.DataFrame(f)


def ler(leitor, meta, idx, cache, device):
    """Uma linha por imagem do conjunto idx: features + y_true, y_pred, erro (+ split/metade/lesão)."""
    t = leitor.transform(cache, idx, device)
    m = meta.iloc[idx].reset_index(drop=True)
    for c in ["image_id", "lesion_id", "metade"]:
        t.insert(0, c, m[c].values)
    t["y_true"] = m.is_malignant.values.astype(int)
    t["y_pred"] = (t.p >= leitor.limiar).astype(int)
    t["erro"] = (t.y_pred != t.y_true).astype(int)
    return t
