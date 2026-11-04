"""
leitor.py — LeitorFuzzy: lê uma rede congelada qualquer por dentro e diz quando ela provavelmente alucinou.

    lf = LeitorFuzzy(rede, tipo=2).fit(dados_referencia, dados_calibracao)
    risco = lf.risco(dados_teste)              # chance de alucinação de cada exemplo
    cls = lf.classificar(dados_teste)          # SEGURO / INCERTO / ALARME, intervalo da nota e o motivo
    tab = lf.diagnostico(dados_teste)          # classes mais difíceis e o porquê

Três zonas: dois limiares escolhidos na calibração (validação cruzada interna, 3 partes), com 95% de confiança:
  SEGURO  = a rede erra no máximo `meta_seguro` (padrão: 1/3 da taxa de erro dela na calibração)
  ALARME  = a rede erra pelo menos `meta_alarme` (padrão: 50%)
  INCERTO = o resto. No tipo 2 a nota é um intervalo: seguro se o TOPO do intervalo está abaixo do limiar,
  alarme se a BASE está acima.

`dados_*` pode ser um DataLoader que devolve (x, y, ...) ou uma tupla de tensores (X, y).
- referência: exemplos do TREINO (sondas, vizinhos). Não precisa de muitos: 2.000 a 5.000 bastam.
- calibração: exemplos que a rede NÃO viu no treino (validação), com rótulo: é onde ela erra de verdade,
  e é com esses erros que o sistema fuzzy aprende.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .blocos import achar_blocos, resumo_vetor
from .fuzzy import FuzzyAditivo, FuzzyAditivo2, TERMOS, limiar_alarme, limiar_seguro, selecionar, zona

EPS = 1e-9
CARACTERISTICAS = ["p", "trocas_opiniao", "prof_decisao", "entropia_probes", "conf_fim", "salto_final",
                   "vistas_mudam", "p_vistas_std", "trajetoria_vistas", "vizinhos_discordam", "dist_vizinhos"]
DE_VISTAS = {"vistas_mudam", "p_vistas_std", "trajetoria_vistas"}
# o que cada característica quer dizer quando AUMENTA o risco
MOTIVO = {"p": "confiança final {t}",
          "trocas_opiniao": "blocos mudaram de opinião",
          "prof_decisao": "decisão só firmou nos últimos blocos",
          "entropia_probes": "blocos indecisos no caminho",
          "conf_fim": "pouca certeza no fim do caminho",
          "salto_final": "certeza apareceu só na saída",
          "vistas_mudam": "resposta muda com a imagem transformada",
          "p_vistas_std": "confiança muda com a imagem transformada",
          "trajetoria_vistas": "caminho muda com a imagem transformada",
          "vizinhos_discordam": "parecidos do treino têm outra resposta",
          "dist_vizinhos": "longe do que a rede viu no treino"}

VISTAS_PADRAO = {"imagem": [lambda x: torch.flip(x, [-1])],                       # espelho horizontal
                 "deslocamento": [lambda x: torch.roll(x, (1, 1), (-2, -1))]}     # dígitos, texto: espelhar muda a classe


def _lotes(dados, lote=256):
    if isinstance(dados, (tuple, list)) and len(dados) >= 1 and torch.is_tensor(dados[0]):
        X = dados[0]; y = dados[1] if len(dados) > 1 else None
        for i in range(0, len(X), lote):
            yield X[i:i + lote], (y[i:i + lote] if y is not None else None)
    else:
        for b in dados:
            if isinstance(b, (tuple, list)):
                yield b[0], (b[1] if len(b) > 1 else None)
            else:
                yield b, None


def _entropia(P):
    P = np.clip(P, EPS, 1)
    return -(P * np.log(P)).sum(1) / np.log(max(P.shape[1], 2))


def _unit(a):
    a = a - a.mean(1, keepdims=True)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)


class LeitorFuzzy:
    def __init__(self, rede, vistas="imagem", blocos=None, max_blocos=20, limiar=None, n_caracteristicas=4,
                 k_vizinhos=15, pca=256, dispositivo=None, tipo=1, meta_seguro=None, meta_alarme=0.5,
                 garantia=True, n_boot=10, semente=0):
        """
        vistas: "imagem" (espelho horizontal), "deslocamento" (1 pixel; para dígitos/texto), uma lista de funções
                x -> x' que NÃO mudam a classe, ou None (uma passada só, sem as características de estabilidade).
                Padrão: 1 versão transformada + a original = 2 passadas.
        blocos: lista de módulos para ler; None = achar sozinho (até `max_blocos`).
        limiar: só no binário; decisão = p >= limiar (padrão: argmax).
        n_caracteristicas: quantas o sistema fuzzy usa (p + n-1 escolhidas).
        tipo: 1 (nota pontual) ou 2 (intervalo; faixa dos termos entre `n_boot` reamostragens da calibração).
        meta_seguro / meta_alarme / garantia: as três zonas (veja o topo do arquivo).
        """
        if tipo not in (1, 2):
            raise ValueError("tipo deve ser 1 ou 2")
        self.tipo, self.meta_seguro, self.meta_alarme, self.garantia = tipo, meta_seguro, meta_alarme, garantia
        self.n_boot, self.semente = n_boot, semente
        self.rede = rede.eval()
        self.disp = dispositivo or next(rede.parameters()).device
        self.vistas = VISTAS_PADRAO[vistas] if isinstance(vistas, str) else list(vistas or [])
        self.blocos_dados, self.max_blocos, self.limiar = blocos, max_blocos, limiar
        self.n_car, self.k, self.pca = n_caracteristicas, k_vizinhos, pca

    # ------------------------------------------------------------------ leitura crua
    @torch.no_grad()
    def _ler(self, dados, vista=None):
        acts = [[] for _ in self.blocos]; cap = [None] * len(self.blocos); pen, probs, ys = [], [], []
        hs = [b.register_forward_hook(lambda m, i, o, k=k: cap.__setitem__(k, resumo_vetor(o).cpu().numpy()))
              for k, b in enumerate(self.blocos)]
        if self.cabeca is not None:
            hs.append(self.cabeca.register_forward_pre_hook(lambda m, i: pen.append(resumo_vetor(i[0]).cpu().numpy())))
        try:
            for x, y in _lotes(dados):
                x = x.to(self.disp)
                s = self.rede(vista(x) if vista else x).float()
                probs.append((torch.cat([1 - torch.sigmoid(s), torch.sigmoid(s)], 1) if s.shape[1] == 1
                              else torch.softmax(s, 1)).cpu().numpy())
                for k in range(len(self.blocos)):
                    acts[k].append(cap[k])
                if self.cabeca is None:
                    pen.append(cap[-1])
                if y is not None:
                    ys.append(np.asarray(y))
        finally:
            for h in hs:
                h.remove()
        A = [np.concatenate(a).astype(np.float32) for a in acts]
        return A, np.concatenate(pen).astype(np.float32), np.concatenate(probs), (np.concatenate(ys).astype(int) if ys else None)

    def _decisao(self, P):
        return (P[:, 1] >= self.limiar).astype(int) if P.shape[1] == 2 and self.limiar is not None else P.argmax(1)

    def _sonda(self, X, y):
        if len(np.unique(y)) < 2:
            return DummyClassifier(strategy="most_frequent").fit(X, y)
        passos = [StandardScaler()]
        n = min(self.pca or 10**9, len(X) - 1)
        if self.pca and X.shape[1] > n:
            passos.append(PCA(n, random_state=0))
        return make_pipeline(*passos, LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")).fit(X, y)

    @staticmethod
    def _prob(s, X, classe):
        P = s.predict_proba(X); col = {c: j for j, c in enumerate(s.classes_)}
        idx = np.array([col.get(c, -1) for c in classe])
        return np.where(idx >= 0, P[np.arange(len(X)), np.maximum(idx, 0)], 0.0), P

    def _knn(self, U, excluir_si=False, bloco=2000):
        idx, dist = [], []; k = min(self.k, len(self.banco) - 1)
        for i in range(0, len(U), bloco):
            S = U[i:i + bloco] @ self.banco.T
            if excluir_si:
                S[np.arange(len(S)), np.arange(i, i + len(S))] = -np.inf
            j = np.argpartition(-S, k, 1)[:, :k]
            idx.append(j); dist.append((1 - np.take_along_axis(S, j, 1)).mean(1))
        return np.concatenate(idx), np.concatenate(dist)

    # ------------------------------------------------------------------ características
    def caracteristicas(self, dados):
        """DataFrame com as 11 características (8 se vistas=None), a decisão, a confiança e o rótulo (se houver)."""
        A, pen, P, y = self._ler(dados)
        dec = self._decisao(P); n = len(dec); r = np.arange(n); conf = P[r, dec]; B = len(self.blocos)
        cc, ents, conc = [], [], []
        for s, a in zip(self.sondas, A):
            c, Ps = self._prob(s, a, dec)
            cc.append(c); ents.append(_entropia(Ps) if Ps.shape[1] > 1 else np.zeros(n)); conc.append(s.classes_[Ps.argmax(1)] == dec)
        cc, conc = np.stack(cc, 1), np.stack(conc, 1).astype(int)
        F = {"p": P[:, 1] if P.shape[1] == 2 else conf,
             "trocas_opiniao": np.abs(np.diff(conc, axis=1)).sum(1),
             "prof_decisao": np.where(conc == 0, np.arange(B)[None] + 1, 0).max(1) / B,
             "entropia_probes": np.stack(ents, 1).mean(1),
             "conf_fim": cc[:, -max(1, B // 3):].mean(1),
             "salto_final": np.abs(conf - cc[:, -1])}
        if self.vistas:
            Pv, T, muda = [conf], [cc], []
            for v in self.vistas:
                Av, _, pv, _ = self._ler(dados, v)
                Pv.append(pv[r, dec]); muda.append(self._decisao(pv) != dec)
                T.append(np.stack([self._prob(s, a, dec)[0] for s, a in zip(self.sondas, Av)], 1))
            F["vistas_mudam"] = np.mean(muda, 0)
            F["p_vistas_std"] = np.stack(Pv, 1).std(1)
            F["trajetoria_vistas"] = np.stack(T, 0).std(0).mean(1)
        idx, dist = self._knn(_unit(pen))
        F["vizinhos_discordam"] = (self.banco_dec[idx] != dec[:, None]).mean(1)
        F["dist_vizinhos"] = dist / self.dist_ref
        F = pd.DataFrame(F)
        F["decisao"], F["confianca"] = dec, conf
        if y is not None:
            F["rotulo"] = y; F["alucinou"] = (dec != y).astype(int)
        return F

    # ------------------------------------------------------------------ treino
    def preparar(self, referencia, log=print):
        """Só a leitura: acha os blocos, treina as sondas e monta o banco de vizinhos (sem o sistema fuzzy).
        Depois disso, `caracteristicas(dados)` já funciona."""
        x0, _ = next(_lotes(referencia, 2))
        if self.blocos_dados is None:
            self.blocos, self.cabeca = achar_blocos(self.rede, x0[:2].to(self.disp), self.max_blocos)
        else:
            self.blocos, self.cabeca = list(self.blocos_dados), None
        log(f"  {len(self.blocos)} pontos de leitura: " + ", ".join(type(b).__name__ for b in self.blocos))
        A, pen, P, y_ref = self._ler(referencia)
        dec = self._decisao(P)
        alvo = dec if len(np.unique(dec)) > 1 else y_ref
        self.sondas = [self._sonda(a, alvo) for a in A]                      # uma sigmoide (sonda) por bloco
        self.banco, self.banco_dec = _unit(pen), dec
        self.dist_ref = float(np.median(self._knn(self.banco, excluir_si=True)[1])) + EPS
        self.n_classes = P.shape[1]
        return self

    def fit(self, referencia, calibracao, log=print):
        """Leitura (`preparar`) + sistema fuzzy aprendido nos erros da rede na calibração."""
        self.preparar(referencia, log)
        return self.ajustar(self.caracteristicas(calibracao), log)

    def ajustar(self, F, log=print):
        """Só o sistema fuzzy e os limiares das zonas, a partir de uma tabela de características já lida
        (`caracteristicas(calibracao)`). Útil para testar tipo 1 × tipo 2 sem ler a rede de novo."""
        if "alucinou" not in F or F.alucinou.nunique() < 2:
            raise ValueError("A calibração precisa de rótulos e de pelo menos um erro e um acerto da rede.")
        y = F.alucinou.values
        fixos = {"p": [0.0, self.limiar if self.limiar is not None else 0.5, 1.0]} if self.n_classes == 2 else None
        cand = [c for c in CARACTERISTICAS if c in F]
        self.escolhidas = selecionar(F, y, cand, self.n_car)
        X = F[self.escolhidas].values
        novo = lambda: (FuzzyAditivo(fixos) if self.tipo == 1 else
                        FuzzyAditivo2(fixos, n_boot=self.n_boot, semente=self.semente))
        # limiares das zonas: notas fora-da-parte DENTRO da calibração (validação cruzada interna)
        from sklearn.model_selection import StratifiedKFold
        lo, hi = np.zeros(len(y)), np.zeros(len(y))
        k = min(3, int(np.bincount(y, minlength=2).min()))
        if k >= 2:
            for a, b in StratifiedKFold(k, shuffle=True, random_state=self.semente).split(X, y):
                lo[b], hi[b], _ = novo().fit(X[a], y[a], self.escolhidas).intervalo(X[b])
        self.fuzzy = novo().fit(X, y, self.escolhidas)
        if k < 2:
            lo, hi, _ = self.fuzzy.intervalo(X)
        meta_seg = self.meta_seguro if self.meta_seguro is not None else y.mean() / 3
        self.t_seguro = limiar_seguro(hi, y, meta_seg, self.garantia)
        self.t_alarme = limiar_alarme(lo, y, self.meta_alarme, self.garantia)
        self.meta_seguro_usada = meta_seg
        self._contrib_media = self.fuzzy.contribuicoes(X).sum(2).mean(0)
        log(f"  tipo {self.tipo} | características: {', '.join(self.escolhidas)} | {int(y.sum())} alucinações na calibração"
            f" | zonas: seguro = erro ≤ {meta_seg:.1%}, alarme = erro ≥ {self.meta_alarme:.0%}")
        return self

    # ------------------------------------------------------------------ uso
    def risco(self, dados, F=None):
        F = self.caracteristicas(dados) if F is None else F
        return self.fuzzy.risco(F[self.escolhidas].values)

    def classificar(self, dados, F=None):
        """Uma linha por exemplo: nota (e o intervalo, no tipo 2), zona SEGURO / INCERTO / ALARME e o motivo."""
        F = self.caracteristicas(dados) if F is None else F
        lo, hi, s = self.fuzzy.intervalo(F[self.escolhidas].values)
        out = pd.DataFrame({"decisao": F.decisao.values, "confianca": F.confianca.values, "risco": s,
                            "risco_inf": lo, "risco_sup": hi, "zona": zona(lo, hi, self.t_seguro, self.t_alarme)})
        out["motivo"] = np.where(out.zona == "seguro", "-", self.explicar(F).motivo.values)   # motivo só quando há aviso
        if "alucinou" in F:
            out["rotulo"], out["alucinou"] = F.rotulo.values, F.alucinou.values
        return out

    def explicar(self, F):
        """Para cada exemplo: a regra que mais aumentou o risco em relação ao exemplo típico, em palavras."""
        X = F[self.escolhidas].values
        Cb = self.fuzzy.contribuicoes(X)                                     # (n, d, 3)
        extra = Cb.sum(2) - self._contrib_media[None]                        # quanto cada característica puxou para cima
        j = extra.argmax(1)
        M = np.stack(self.fuzzy.part.pertinencias(X), 1)                     # termo mais ativo daquela característica
        k = M[np.arange(len(X)), j].argmax(1)
        nomes = [self.escolhidas[a] for a in j]
        return pd.DataFrame({"caracteristica": nomes, "termo": [TERMOS[b] for b in k],
                             "motivo": [MOTIVO.get(n, n).format(t=TERMOS[b].lower()) for n, b in zip(nomes, k)]})

    def diagnostico(self, dados, nomes_classes=None):
        """Uma linha por classe verdadeira: acurácia, alucinações, para qual classe ela mais erra,
        quantas alucinações o sistema avisou (alarme ou incerto), quantas escaparam (seguro) e o motivo mais comum."""
        F = self.caracteristicas(dados)
        if "rotulo" not in F:
            raise ValueError("O diagnóstico por classe precisa dos rótulos.")
        Z = self.classificar(None, F); Z["motivo"] = self.explicar(F).motivo.values   # aqui: motivo de todo erro
        F = pd.concat([F, Z[["risco", "risco_inf", "risco_sup", "zona", "motivo"]]], axis=1)
        nome = (lambda c: nomes_classes[c]) if nomes_classes is not None else (lambda c: str(c))
        lin = []; geral = F[F.alucinou == 1].motivo.value_counts(normalize=True)
        for c, g in F.groupby("rotulo"):
            e = g[g.alucinou == 1]
            linha = {"classe": nome(c), "exemplos": len(g), "acurácia": round(1 - g.alucinou.mean(), 3),
                     "alucinações": len(e), "no alarme": int((e.zona == "alarme").sum()),
                     "no incerto": int((e.zona == "incerto").sum()), "escaparam (seguro)": int((e.zona == "seguro").sum()),
                     "confunde mais com": nome(e.decisao.mode()[0]) if len(e) else "-"}
            if len(e):
                mot = e.motivo.value_counts(normalize=True)
                linha["motivos (% dos erros)"] = "; ".join(f"{m} {100 * v:.0f}%" for m, v in mot.head(2).items())
                lift = (mot / geral.reindex(mot.index)).where(e.motivo.value_counts() >= 3)   # mais comum AQUI do que no geral
                linha["típico desta classe"] = lift.idxmax() if lift.notna().any() and lift.max() > 1.25 else "-"
            else:
                linha["motivos (% dos erros)"] = linha["típico desta classe"] = "-"
            lin.append(linha)
        self.ultimo_detalhe = F
        return pd.DataFrame(lin).sort_values("acurácia").reset_index(drop=True)

    def resumo_zonas(self, dados=None, Z=None):
        """Tabela das três zonas: % dos exemplos, taxa de erro da rede em cada uma e % dos erros que caem nela."""
        Z = self.classificar(dados) if Z is None else Z
        if "alucinou" not in Z:
            raise ValueError("O resumo precisa dos rótulos.")
        y = Z.alucinou.values; lin = []
        for nz in ("seguro", "incerto", "alarme"):
            m = (Z.zona == nz).values
            lin.append({"zona": nz, "% dos exemplos": m.mean(), "taxa de erro da rede": y[m].mean() if m.any() else np.nan,
                        "% dos erros": y[m].sum() / max(y.sum(), 1)})
        return pd.DataFrame(lin)
