"""
leitura_rede.py — features do COMPORTAMENTO INTERNO da rede (nada medido nos pixels da imagem).

Três eixos de informação, escolhidos pela análise das rodadas 4 e 5:

1) A REDE MUDA DE IDEIA AO LONGO DAS CAMADAS?
   Em cada sub-bloco, um classificador simples (probe) diz o que a rede responderia se parasse ali.
   Os probes imitam a DECISÃO FINAL da rede, então a trajetória mostra quando ela se compromete.
     trocas_opiniao   quantas vezes a opinião troca entre maligno e benigno ao longo das camadas
     prof_decisao     a partir de que fração da rede a opinião não muda mais (0 = cedo, 1 = só no fim)
     entropia_probes  indecisão média das camadas
     conf_fim         o quanto as últimas camadas já concordam com a resposta final
     salto_final      a cabeça da rede discorda do último bloco

2) A REDE MUDA DE IDEIA ENTRE VERSÕES DA MESMA IMAGEM? (espelhada / invertida — a lesão é a mesma)
     vistas_mudam     fração das versões em que a decisão final muda
     p_vistas_std     quanto P varia entre as versões
     trajetoria_vistas quanto a trajetória de opinião nas camadas varia entre as versões

3) O QUE A REDE ACHA DE CASOS PARECIDOS DO TREINO? (vizinhos no espaço interno do último bloco)
     vizinhos_discordam  fração dos vizinhos para os quais a rede deu a resposta oposta
     dist_vizinhos       quão longe estão os vizinhos (1 = típico)

   + prox_p: proximidade de P ao limiar (1 = em cima do limiar). É a referência a superar.

Também devolve a trajetória crua (probe_b0..probe_bN) para análise.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

EPS = 1e-9
FEATURES = ["trocas_opiniao", "prof_decisao", "entropia_probes", "conf_fim", "salto_final",
            "vistas_mudam", "p_vistas_std", "trajetoria_vistas",
            "vizinhos_discordam", "dist_vizinhos", "prox_p"]
VISTAS = {"original": lambda x: x, "espelho_h": lambda x: torch.flip(x, [3]),
          "espelho_v": lambda x: torch.flip(x, [2]), "giro_180": lambda x: torch.flip(x, [2, 3])}


def blocos_efficientnet(model, finos=True):
    """finos=True: os 18 sub-blocos do EfficientNet-B0; False: os 9 estágios."""
    if not finos:
        return list(model.features.children())
    out = [model.features[0]]
    for estagio in list(model.features.children())[1:-1]:
        out += list(estagio.children())
    return out + [model.features[-1]]


class LeitorRede:
    def __init__(self, model, limiar=0.5, blocos=None, k_vizinhos=15):
        self.model, self.limiar, self.k = model, limiar, k_vizinhos
        self.blocos = list(blocos) if blocos is not None else list(model.features.children())
        self.camada_final = [m for m in model.classifier.modules() if isinstance(m, nn.Linear)][-1]
        self.B = len(self.blocos)

    @torch.no_grad()
    def _ler(self, lotes, vista=lambda x: x):
        """Ativações médias por bloco e P, para cada imagem (aplicando a 'vista' antes)."""
        self.model.eval()
        acum, P = [[] for _ in range(self.B)], []
        for x in lotes:
            cap = [None] * self.B
            hs = [b.register_forward_hook(lambda m, i, o, k=k: cap.__setitem__(k, o.mean((2, 3)).cpu().numpy()))
                  for k, b in enumerate(self.blocos)]
            try:
                P.append(self.model(vista(x)).view(-1).cpu().numpy())
            finally:
                for h in hs: h.remove()
            for k in range(self.B): acum[k].append(cap[k])
        return [np.concatenate(a).astype(np.float64) for a in acum], np.concatenate(P)

    def fit(self, lotes, rotulos):
        """Referência = imagens que a rede viu no treino."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        A, P = self._ler(lotes)
        dec = (P >= self.limiar).astype(int)
        alvo = dec if len(np.unique(dec)) == 2 else np.asarray(rotulos).astype(int)
        self.probes = [make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
                       .fit(a, alvo) for a in A]
        self.banco = self._unit(A[-1]); self.banco_dec = dec                      # vizinhos: último bloco
        d = self._knn_dist(self.banco, excluir_si=True)
        self.dist_ref = float(np.median(d)) + EPS
        return self

    @staticmethod
    def _unit(a):
        a = a - a.mean(1, keepdims=True)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)

    def _knn(self, U, excluir_si=False):
        S = U @ self.banco.T
        if excluir_si:
            np.fill_diagonal(S, -np.inf)
        return np.argsort(-S, 1)[:, :self.k], S

    def _knn_dist(self, U, excluir_si=False):
        idx, S = self._knn(U, excluir_si)
        return (1 - np.take_along_axis(S, idx, 1)).mean(1)

    def _trajetoria(self, A):
        return np.stack([pr.predict_proba(a)[:, 1] for pr, a in zip(self.probes, A)], 1)   # (n, B)

    def transform(self, fazer_lotes) -> pd.DataFrame:
        """fazer_lotes: função sem argumentos que devolve um gerador novo de lotes (lido uma vez por vista)."""
        res = {nome: self._ler(fazer_lotes(), v) for nome, v in VISTAS.items()}
        A, P = res["original"]
        pp = self._trajetoria(A)
        final = (P >= self.limiar).astype(int)
        f = {"p": P}
        # 1) ao longo das camadas
        op = (pp >= 0.5).astype(int)
        f["trocas_opiniao"] = np.abs(np.diff(op, axis=1)).sum(1)
        f["prof_decisao"] = np.where(op != final[:, None], np.arange(self.B)[None, :] + 1, 0).max(1) / self.B
        f["entropia_probes"] = -(pp * np.log(pp + EPS) + (1 - pp) * np.log(1 - pp + EPS)).mean(1)
        conf = np.where(final[:, None] == 1, pp, 1 - pp)
        f["conf_fim"] = conf[:, -max(1, self.B // 3):].mean(1)
        f["salto_final"] = np.abs(P - pp[:, -1])
        # 2) entre versões da mesma imagem
        Ps = np.stack([res[v][1] for v in VISTAS], 1)                                         # (n, V)
        f["vistas_mudam"] = ((Ps[:, 1:] >= self.limiar).astype(int) != final[:, None]).mean(1)
        f["p_vistas_std"] = Ps.std(1)
        T = np.stack([self._trajetoria(res[v][0]) for v in VISTAS], 0)                        # (V, n, B)
        f["trajetoria_vistas"] = T.std(0).mean(1)
        # 3) vizinhos no espaço interno
        idx, S = self._knn(self._unit(A[-1]))
        f["vizinhos_discordam"] = (self.banco_dec[idx] != final[:, None]).mean(1)
        f["dist_vizinhos"] = (1 - np.take_along_axis(S, idx, 1)).mean(1) / self.dist_ref
        # referência
        f["prox_p"] = 1 - np.abs(P - self.limiar) / max(self.limiar, 1 - self.limiar)
        for k in range(self.B):
            f[f"probe_b{k}"] = pp[:, k]
        return pd.DataFrame(f)


def lotes_de_imagens(caminhos, tfm, device, bs=32):
    caminhos = list(caminhos)
    for i in range(0, len(caminhos), bs):
        yield torch.stack([tfm(Image.open(c).convert("RGB")) for c in caminhos[i:i + bs]]).to(device)


def ler(leitor, df, tfm, device, bs=32):
    """Uma linha por imagem: features + y_true, y_pred, erro."""
    t = leitor.transform(lambda: lotes_de_imagens(df.image_path, tfm, device, bs))
    t.insert(0, "image_id", df.image_id.values)
    t["y_true"] = df.is_malignant.values.astype(int)
    t["y_pred"] = (t.p >= leitor.limiar).astype(int)
    t["erro"] = (t.y_pred != t.y_true).astype(int)
    return t
