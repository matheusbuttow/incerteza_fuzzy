"""
analise_correcao.py — melhoria 5 e piloto da "segunda opinião": além de AVISAR (zona ALARME), dá para CORRIGIR a resposta?
E uma integral de Choquet sobre a profundidade da rede funciona no lugar da softmax?

Base: a rede decora os rótulos errados nas últimas camadas; os blocos iniciais quase não são afetados (Stephenson et al., ICLR 2021).
Usa o que o rodar.py salvou: extras.npz["probs_blocos"] (classe que uma sonda treinada com o rótulo aponta em cada bloco),
extras.npz["probs"] (a softmax final), features.csv.gz (decisão e rótulo) e zonas_por_imagem.csv.gz (as zonas do sistema).
Rodadas antigas ganham probs_blocos rodando de novo o rodar.py/rodar_3dias.sh (só relê a rede; não retreina).

Respostas comparadas:
  rede                          a resposta da rede (softmax)
  média dos blocos iniciais     média das sondas dos 2/3 iniciais da rede (sem treino)
  média de todos os blocos      média de todas as sondas + a softmax (o "Early Exit Ensembles" de Qendro et al., 2021, sem fuzzy)
  Choquet da profundidade       a rede é dividida em G trechos (os blocos de cada trecho viram a média de suas sondas; a softmax
                                é o último trecho); uma Choquet 2-aditiva, a mesma para todas as classes, junta os trechos e dá a
                                nota de cada classe. Pesos ≥ 0 que somam 1 (forma min/max, como no CHOQOLATE), aprendidos com os
                                rótulos do conjunto de calibração, por validação cruzada como o resto do sistema.
Cada uma é avaliada de dois jeitos:
  "no lugar da softmax"  vale para todas as imagens
  "só no ALARME"         o monitor fuzzy decide onde trocar: fora do ALARME fica a resposta da rede

    python analise_correcao.py                       # → resultados/correcao.xlsx
    python analise_correcao.py --exp cifar10n_worst --pasta resultados_teste_rapido
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

import config as C
from analise_geral import rodadas

G_TRECHOS = 6


def trechos(PB, P, G=G_TRECHOS):
    """(n, B, K) sondas + (n, K) softmax → (n, G, K): os blocos em G−1 trechos de profundidade (média) e a softmax como o último."""
    B = PB.shape[1]; g = max(1, min(G - 1, B))
    cortes = np.array_split(np.arange(B), g)
    return np.stack([PB[:, c].mean(1) for c in cortes] + [P], 1).astype(np.float32)


class ChoquetProfundidade(torch.nn.Module):
    """Choquet 2-aditiva na forma de Grabisch: Σ a_j u_j + Σ_{j<l} [b_jl min(u_j, u_l) + c_jl max(u_j, u_l)], com todos os pesos ≥ 0
    somando 1 (softmax dos parâmetros) — isso garante uma medida fuzzy monótona e normalizada. A mesma medida para todas as classes."""

    def __init__(self, G):
        super().__init__()
        self.G = G; self.pares = list(combinations(range(G), 2))
        self.w = torch.nn.Parameter(torch.zeros(G + 2 * len(self.pares)))
        self.log_t = torch.nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))   # temperatura das notas

    def pesos(self):
        return torch.softmax(self.w, 0)

    def forward(self, U):                                     # U: (n, G, K) → notas (n, K)
        w = self.pesos(); G, pr = self.G, self.pares
        i, j = torch.tensor([p[0] for p in pr]), torch.tensor([p[1] for p in pr])
        lin = torch.einsum("g,ngk->nk", w[:G], U)
        mn = torch.einsum("p,npk->nk", w[G:G + len(pr)], torch.minimum(U[:, i], U[:, j]))
        mx = torch.einsum("p,npk->nk", w[G + len(pr):], torch.maximum(U[:, i], U[:, j]))
        return lin + mn + mx

    def shapley(self):
        """Valor de Shapley de cada trecho (soma 1). Para a forma min/max: a_j + metade de cada b e c dos pares que contêm j."""
        w = self.pesos().detach().numpy(); G, pr = self.G, self.pares
        s = w[:G].copy()
        for t, (a, b) in enumerate(pr):
            for v in (w[G + t], w[G + len(pr) + t]):
                s[a] += v / 2; s[b] += v / 2
        return s

    def interacoes(self):
        """Índice de interação de cada par de trechos: + = complementares (min, um E), − = substitutos (max, um OU)."""
        w = self.pesos().detach().numpy(); G, pr = self.G, self.pares
        return {p: w[G + t] - w[G + len(pr) + t] for t, p in enumerate(pr)}


def ajustar(U, y, passos=400, semente=0):
    torch.manual_seed(semente)
    m = ChoquetProfundidade(U.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=0.05)
    Ut, yt = torch.as_tensor(U), torch.as_tensor(y, dtype=torch.long)
    for _ in range(passos):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(Ut) * torch.exp(m.log_t), yt)
        loss.backward(); opt.step()
    return m


def prever(m, U):
    with torch.no_grad():
        return torch.softmax(m(torch.as_tensor(U)) * torch.exp(m.log_t), 1).numpy()


def ece_topo(p, acerto, bins=15):
    """ECE da confiança na classe escolhida (a calibração usual da saída)."""
    c = p.max(1); b = np.minimum((c * bins).astype(int), bins - 1)
    return sum(abs(acerto[b == k].mean() - c[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any())


def uma_rodada(exp, sem, d):
    ext = np.load(d / "extras.npz")
    if "probs_blocos" not in ext.files:
        return None, None
    F = pd.read_csv(d / "features.csv.gz"); y = F.rotulo.values.astype(int); dec = F.decisao.values.astype(int)
    P = ext["probs"].astype(np.float32); PB = ext["probs_blocos"].astype(np.float32); B = PB.shape[1]
    alarme = pd.read_csv(d / "zonas_por_imagem.csv.gz")["zona Fuzzy aditivo"].astype(str).str.lower().eq("alarme").values
    U = trechos(PB, P)
    ini = PB[:, :max(1, int(np.ceil(2 * B / 3)))].mean(1)
    resp = {"rede": dec, "média dos blocos iniciais": ini.argmax(1),
            "média de todos os blocos": np.concatenate([PB, P[:, None]], 1).mean(1).argmax(1)}
    Pch = np.zeros_like(P); SH = []
    for tr, te in StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=sem).split(U, F.erro.values):
        m = ajustar(U[tr], y[tr], semente=sem); Pch[te] = prever(m, U[te]); SH.append(m.shapley())
    resp["Choquet da profundidade"] = Pch.argmax(1)

    ds, arq = C.GRADE_INFO.get(exp, (exp, "?"))
    lin = {"experimento": exp, "dataset": ds, "rede": arq, "semente": sem, "blocos": B, "imagens": len(y),
           "acurácia da rede": (dec == y).mean(), "imagens no ALARME": int(alarme.sum()),
           "erros da rede no ALARME": int(((dec != y) & alarme).sum()), "erros da rede fora do ALARME": int(((dec != y) & ~alarme).sum())}
    for nome, r in resp.items():
        if nome == "rede":
            continue
        lin[f"{nome}: no lugar da softmax"] = (r == y).mean()
        lin[f"{nome}: só no ALARME"] = (np.where(alarme, r, dec) == y).mean()
        if alarme.any():
            lin[f"{nome}: acerto dentro do ALARME"] = (r[alarme] == y[alarme]).mean()
            lin[f"{nome}: erros corrigidos no ALARME"] = int(((dec != y) & (r == y) & alarme).sum())
            lin[f"{nome}: acertos estragados no ALARME"] = int(((dec == y) & (r != y) & alarme).sum())
    if alarme.any():
        lin["rede: acerto dentro do ALARME"] = (dec[alarme] == y[alarme]).mean()
        lin["teto: algum bloco acerta (ALARME)"] = (PB[alarme].argmax(2) == y[alarme, None]).any(1).mean()
    lin["ECE da softmax"] = ece_topo(P, dec == y)
    lin["ECE da Choquet da profundidade"] = ece_topo(Pch, Pch.argmax(1) == y)
    sh = np.mean(SH, 0)
    for g, v in enumerate(sh):
        lin[f"Shapley trecho {g + 1}" + (" (softmax)" if g == len(sh) - 1 else "")] = v
    return lin, sh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", default=str(C.SAIDA))
    ap.add_argument("--exp", nargs="*")
    a = ap.parse_args()
    pasta = Path(a.pasta)
    lista = [r for r in rodadas(pasta) if not a.exp or r[0] in a.exp]
    L, falta = [], 0
    for exp, sem, d in lista:
        try:
            lin, _ = uma_rodada(exp, sem, d)
        except Exception as err:
            print(f"  *** {exp} semente {sem}: {type(err).__name__}: {err}"); continue
        if lin is None:
            falta += 1; continue
        L.append(lin)
        print(f"  {exp:28s} {sem:>5}: rede {lin['acurácia da rede']:.3f} | Choquet no lugar da softmax "
              f"{lin['Choquet da profundidade: no lugar da softmax']:.3f} | só no ALARME {lin['Choquet da profundidade: só no ALARME']:.3f}", flush=True)
    if falta:
        print(f"\n{falta} rodadas ainda sem as classes por bloco: rode de novo o rodar_3dias.sh (só relê a rede; não retreina).")
    if not L:
        raise SystemExit("Nenhuma rodada com probs_blocos.")
    L = pd.DataFrame(L)
    cols = [c for c in L.columns if c.startswith(("acurácia", "média", "Choquet", "rede:", "teto", "ECE", "Shapley", "erros", "imagens no"))]
    cel = L.groupby(["dataset", "rede"])[cols].mean()
    # ganhos em pontos percentuais de acurácia, contra a rede
    resumo = []
    for nome in ["média dos blocos iniciais", "média de todos os blocos", "Choquet da profundidade"]:
        for modo in ["no lugar da softmax", "só no ALARME"]:
            g = 100 * (L[f"{nome}: {modo}"] - L["acurácia da rede"])
            resumo.append({"resposta": nome, "onde": modo, "ganho médio (pontos %)": g.mean(), "melhor": g.max(), "pior": g.min(),
                           "rodadas que melhoram": int((g > 0).sum()), "rodadas": len(g)})
    resumo = pd.DataFrame(resumo)
    arq = pasta / "correcao.xlsx"
    with pd.ExcelWriter(arq) as xw:
        resumo.round(3).to_excel(xw, sheet_name="resumo", index=False)
        cel.round(4).to_excel(xw, sheet_name="por_combinacao")
        L.round(4).to_excel(xw, sheet_name="por_rodada", index=False)
    pd.set_option("display.width", 220)
    print("\nGANHO DE ACURÁCIA SOBRE A REDE (pontos percentuais):\n" + resumo.round(2).to_string(index=False))
    sh = L[[c for c in L.columns if c.startswith("Shapley")]].mean()
    print("\nPESO DE CADA TRECHO DE PROFUNDIDADE NA CHOQUET (Shapley médio):\n" + sh.round(3).to_string())
    print(f"\nECE: softmax {L['ECE da softmax'].mean():.3f} × Choquet da profundidade {L['ECE da Choquet da profundidade'].mean():.3f}")
    print(f"\nPlanilha: {arq}")


if __name__ == "__main__":
    main()
