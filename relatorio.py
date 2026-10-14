"""
relatorio.py — junta os resultados de todos os experimentos e sementes e gera:
  resultados/relatorio.xlsx      todas as tabelas (média ± desvio entre sementes e por semente)
  resultados/figuras/*.png       todos os gráficos
  resultados/relatorio.html      página única com tabelas e gráficos, para abrir no navegador

Uso:  python relatorio.py            (ou  python relatorio.py --teste-rapido)
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

import config as C

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
NOSSO = {"NOSSO: Wang-Mendel": "#2a78d6", "NOSSO: Fuzzy aditivo": "#eb6834"}
CHAVE = {"Só P": ("#7a7a7a", "--"), "Softmax (MSP)": ("#6250d6", ":"), "MC Dropout (desvio)": ("#e87ba4", "-"),
         "Chen et al. 2019": ("#1baf7a", "-"), "ConfidNet": ("#eda100", "-"),
         "NOSSO: Wang-Mendel": ("#2a78d6", "-"), "NOSSO: Fuzzy aditivo": ("#eb6834", "-")}
METRICAS = ["AUROC", "AUPR-erro", "FPR@95%TPR", "E-AURC (x1000)", "acurácia encaminhando 10%", "acurácia encaminhando 20%",
            "acurácia encaminhando 30%"]

# Números PUBLICADOS por cada artigo (com a rede e os dados deles). Compare só o GANHO sobre o softmax de cada um.
PUBLICADOS = [  # (artigo, cenário, nosso experimento equivalente, softmax do artigo, método do artigo)
    ("Chen et al. 2019", "CIFAR-10, ResNet", "cifar10", 0.91, 0.91),
    ("Chen et al. 2019", "CIFAR-10 ruído 30%, ResNet", "cifar10_ruido30", 0.74, 0.88),
    ("ConfidNet 2019", "MNIST, MLP", "mnist", 0.971, 0.978),
    ("ConfidNet 2019", "SVHN, CNN pequena", "svhn", 0.932, 0.934),
    ("ConfidNet 2019", "CIFAR-10, VGG-16", "cifar10_vgg16", 0.915, 0.921),
    ("ConfidNet 2019", "CIFAR-100, VGG-16", "cifar100_vgg16", 0.857, 0.863),
    ("FD-Shifts 2023", "CIFAR-10, VGG-13", "cifar10_vgg16", 0.937, np.nan),
    ("FD-Shifts 2023", "CIFAR-100, VGG-13", "cifar100_vgg16", 0.873, np.nan),
    ("FD-Shifts 2023", "SVHN, CNN pequena", "svhn", 0.929, np.nan),
]


def ler_tudo(saida):
    def junta(nome, leitor=pd.read_csv):
        partes = []
        for p in sorted(saida.glob(f"*/semente_*/{nome}")):
            d = leitor(p); d.insert(0, "semente", int(p.parent.name.split("_")[1])); d.insert(0, "experimento", p.parent.parent.name)
            partes.append(d)
        return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    met = pd.concat([pd.read_csv(p) for p in sorted(saida.glob("*/semente_*/metricas.csv"))], ignore_index=True)
    redes = pd.DataFrame([json.loads(p.read_text()) for p in sorted(saida.glob("*/semente_*/info_rede.json"))])
    est = pd.DataFrame([{"experimento": p.parent.parent.name, "semente": int(p.parent.name.split("_")[1]),
                         **{k: v for k, v in json.loads(p.read_text()).items() if k != "escolhas por fold"}}
                        for p in sorted(saida.glob("*/semente_*/regras_info.json"))])
    return {"metricas": met, "redes": redes, "estabilidade": est, "diagnostico": junta("diagnostico_features.csv"),
            "rejeicao": junta("curva_rejeicao.csv"), "risco": junta("curva_risco_cobertura.csv"), "roc": junta("curva_roc.csv"),
            "regras_wm": junta("regras_wang_mendel.csv"), "regras_ad": junta("regras_fuzzy_aditivo.csv"),
            "dx_type": junta("ham_dx_type.csv"), "multifotos": junta("ham_multifotos.csv"), "cifar10h": junta("cifar10h.csv")}


def agregar(met):
    g = met.groupby(["experimento", "método"], sort=False)
    ag = g[METRICAS].agg(["mean", "std"])
    ag.columns = [f"{a} ({'média' if b == 'mean' else 'desvio'})" for a, b in ag.columns]
    ag["sementes"] = g.size()
    ag["sementes com ganho significativo sobre só P"] = g["IC95 ganho inf"].apply(lambda s: int((s > 0).sum()))
    ag["ganho AUROC sobre só P (média)"] = g["ganho AUROC sobre só P"].mean()
    return ag.reset_index()


def tabela_geral(ag):
    t = ag.assign(v=ag["AUROC (média)"].map("{:.3f}".format) + " ± " + ag["AUROC (desvio)"].fillna(0).map("{:.3f}".format))
    return t.pivot(index="método", columns="experimento", values="v").reindex(ag["método"].unique())


def publicados(ag):
    m = ag.set_index(["experimento", "método"])
    lin = []
    for art, cen, exp, sm, metodo in PUBLICADOS:
        lin.append({"artigo": art, "cenário do artigo": cen, "softmax (artigo)": sm, "método do artigo": metodo,
                    "ganho do artigo": metodo - sm if not np.isnan(metodo) else np.nan, "nosso experimento": exp})
        if (exp, "Softmax (MSP)") in m.index:
            sm_n = m.loc[(exp, "Softmax (MSP)"), "AUROC (média)"]
            melhor = max(m.loc[(exp, k), "AUROC (média)"] for k in NOSSO if (exp, k) in m.index)
            lin[-1].update({"softmax (nosso)": sm_n, "nosso melhor (fuzzy)": melhor, "nosso ganho": melhor - sm_n})
    return pd.DataFrame(lin)


# ----------------------------------------------------------------------------- gráficos
def _salvar(fig, pasta, nome, figs):
    fig.tight_layout(); fig.savefig(pasta / nome, bbox_inches="tight"); plt.close(fig); figs.append(nome)


def graficos(R, ag, pasta, saida):
    figs = []; exps = list(dict.fromkeys(ag.experimento))
    # visão geral: métodos × experimentos
    t = ag.pivot(index="método", columns="experimento", values="AUROC (média)").reindex(ag["método"].unique())[exps]
    fig, ax = plt.subplots(figsize=(1.3 * len(exps) + 4, 0.38 * len(t) + 1.5))
    im = ax.imshow(t.values, cmap="Blues", aspect="auto", vmin=np.nanmin(t.values), vmax=np.nanmax(t.values))
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if v > np.nanpercentile(t.values, 70) else "black",
                        fontweight="bold" if t.index[i].startswith("NOSSO") else "normal")
    ax.set_xticks(range(len(exps)), exps, rotation=30, ha="right"); ax.set_yticks(range(len(t)), t.index)
    ax.set_title("AUROC para detectar os erros da rede (média entre sementes)"); fig.colorbar(im, ax=ax, shrink=0.6)
    _salvar(fig, pasta, "00_visao_geral_auroc.png", figs)

    for exp in exps:
        a = ag[ag.experimento == exp].sort_values("AUROC (média)")
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(a) + 1.2))
        cores = [NOSSO.get(m, "#b4b2a9") for m in a["método"]]
        ax.barh(a["método"], a["AUROC (média)"], xerr=a["AUROC (desvio)"].fillna(0), color=cores, height=0.6, capsize=2)
        base = a.set_index("método").loc["Só P", "AUROC (média)"]
        ax.axvline(base, color="#555", ls="--", lw=1); ax.text(base, len(a) - 0.3, " só P", fontsize=8, color="#555")
        lo = max(0.4, a["AUROC (média)"].min() - 0.05); ax.set_xlim(lo, min(1.0, a["AUROC (média)"].max() + 0.04))
        ax.set_xlabel("AUROC (detectar os erros da rede) — média ± desvio entre sementes"); ax.set_title(f"{exp}: todos os métodos")
        _salvar(fig, pasta, f"01_auroc_{exp}.png", figs)

        for nome_df, arq, x, yv, tit, xl, yl in [("rejeicao", "02_rejeicao", "% encaminhado", "acurácia nas que ficam",
                                                  "acurácia ao encaminhar as mais incertas ao especialista", "% encaminhado (os mais incertos)", "acurácia nas imagens que ficam"),
                                                 ("risco", "03_risco_cobertura", "cobertura", "risco", "curva risco-cobertura (menor = melhor)",
                                                  "cobertura (fração que a rede decide sozinha)", "taxa de erro nas que ficam")]:
            d = R[nome_df]; d = d[d.experimento == exp]
            if d.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.2))
            for m, (cor, ls) in CHAVE.items():
                dm = d[d["método"] == m]
                if dm.empty:
                    continue
                s = dm.groupby(x)[yv].agg(["mean", "min", "max"])
                ax.plot(s.index, s["mean"], color=cor, ls=ls, lw=2.4 if m in NOSSO else 1.5, label=m)
                ax.fill_between(s.index, s["min"], s["max"], color=cor, alpha=0.08)
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{exp}: {tit}"); ax.legend(frameon=False, fontsize=8)
            _salvar(fig, pasta, f"{arq}_{exp}.png", figs)

        d = R["roc"]; d = d[(d.experimento == exp) & (d.semente == d[d.experimento == exp].semente.min())]
        if not d.empty:
            fig, ax = plt.subplots(figsize=(4.8, 4.5))
            for m, (cor, ls) in CHAVE.items():
                dm = d[d["método"] == m]
                if len(dm):
                    ax.plot(dm.FPR, dm.TPR, color=cor, ls=ls, lw=2.2 if m in NOSSO else 1.3, label=m)
            ax.plot([0, 1], [0, 1], color="#ccc", lw=0.8); ax.set_xlabel("acertos marcados como dúvida"); ax.set_ylabel("erros encontrados")
            ax.set_title(f"{exp}: curva ROC (1ª semente)"); ax.legend(frameon=False, fontsize=7)
            _salvar(fig, pasta, f"04_roc_{exp}.png", figs)

        p = sorted((saida / exp).glob("semente_*/features.csv.gz"))
        if p:
            tab = pd.read_csv(p[0]); cols = [c for c in tab.columns if c.startswith("probe_b")]
            PB = tab[cols].values; B = len(cols)
            fig, ax = plt.subplots(figsize=(7, 3.6))
            for nome, msk, cor in [("acertos", tab.erro == 0, "#2a78d6"), ("erros", tab.erro == 1, "#eb6834")]:
                if msk.sum():
                    q = np.percentile(PB[msk.values], [25, 50, 75], axis=0)
                    ax.plot(range(1, B + 1), q[1], color=cor, lw=2, label=f"{nome} (n={msk.sum()})")
                    ax.fill_between(range(1, B + 1), q[0], q[2], color=cor, alpha=0.15)
            ax.axhline(0.5, color="gray", ls="--", lw=1); ax.set_ylim(0, 1); ax.set_xlabel("bloco da rede")
            ax.set_ylabel("concorda com a resposta final"); ax.set_title(f"{exp}: como a opinião evolui nas camadas"); ax.legend(frameon=False)
            _salvar(fig, pasta, f"05_curva_opiniao_{exp}.png", figs)
            if exp == "ham10000" and "image_path" in tab:
                ex = [("Acerto tranquilo", tab[tab.erro == 0].entropia_probes.idxmin()), ("Erro hesitante", tab[tab.erro == 1].trocas_opiniao.idxmax()),
                      ("Erro confiante", tab[tab.erro == 1].prox_p.idxmin())] if tab.erro.sum() else []
                if ex:
                    fig, axs = plt.subplots(len(ex), 2, figsize=(9, 3.1 * len(ex)), gridspec_kw={"width_ratios": [1, 1.6]})
                    for (nome, i), (a1, a2) in zip(ex, axs):
                        r = tab.loc[i]
                        a1.imshow(Image.open(r.image_path).convert("RGB")); a1.axis("off")
                        a1.set_title(f"{nome}\n{r.image_id} | real: {'maligna' if r.rotulo else 'benigna'} | rede: {'maligna' if r.decisao else 'benigna'} (P={r.p:.2f})",
                                     fontsize=8, loc="left")
                        a2.plot(range(1, B + 1), PB[i], marker="o", color="#b8325a" if r.erro else "#2a78d6")
                        a2.axhline(0.5, color="gray", ls="--", lw=1); a2.set_ylim(0, 1); a2.set_xlabel("sub-bloco"); a2.set_ylabel("concorda com a resposta")
                        a2.set_title(f"mudou de ideia {int(r.trocas_opiniao)}x | hesitação {r.entropia_probes:.2f} | salto final {r.salto_final:.2f}", fontsize=8, loc="left")
                    _salvar(fig, pasta, "10_exemplos_ham.png", figs)

        rw = R["regras_wm"]; rw = rw[(rw.experimento == exp) & (rw.semente == rw[rw.experimento == exp].semente.min())]
        if len(rw):
            top = rw.assign(peso=rw["certeza (CF)"] * rw["suporte"]).nlargest(10, "peso").iloc[::-1]
            fig, ax = plt.subplots(figsize=(10, 0.42 * len(top) + 1))
            ax.barh(range(len(top)), top["certeza (CF)"], color=["#eb6834" if c == "ERRO" else "#2a78d6" for c in top["conclusão"]], height=0.6)
            ax.set_yticks(range(len(top)), [f"{r} (suporte {s:.0%})" for r, s in zip(top.regra, top.suporte)], fontsize=7.5)
            ax.set_xlim(0, 1); ax.set_xlabel("certeza da regra (CF)"); ax.set_title(f"{exp}: principais regras Wang-Mendel (azul = ACERTO, laranja = ERRO)")
            _salvar(fig, pasta, f"08_regras_wm_{exp}.png", figs)

    m = R["metricas"]; nossos = m[m["método"].isin(list(NOSSO) + ["Chen et al. 2019"])]
    if len(nossos):
        lin = nossos.sort_values(["experimento", "método", "semente"]).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(lin) + 1.2))
        for i, r in lin.iterrows():
            cor = NOSSO.get(r["método"], "#1baf7a")
            ax.plot([r["IC95 ganho inf"], r["IC95 ganho sup"]], [i, i], color=cor, lw=1.5)
            ax.plot(r["ganho AUROC sobre só P"], i, "o", color=cor, ms=4)
        ax.axvline(0, color="#555", lw=1, ls="--")
        ax.set_yticks(range(len(lin)), [f"{r.experimento} | {r['método'].replace('NOSSO: ', '')} | s{r.semente}" for _, r in lin.iterrows()], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("ganho de AUROC sobre só P (IC 95%) — à direita do zero = melhor que P")
        ax.set_title("O ganho é significativo em cada semente?")
        _salvar(fig, pasta, "06_ganho_sobre_P.png", figs)

    dg = R["diagnostico"]
    if len(dg):
        h = dg.groupby(["feature", "experimento"])["AUROC além de P"].mean().unstack()
        h = h.loc[h.mean(1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(1.2 * h.shape[1] + 4, 0.35 * len(h) + 1.5))
        im = ax.imshow(h.values, cmap="Oranges", aspect="auto", vmin=0.5)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                if not np.isnan(h.values[i, j]):
                    ax.text(j, i, f"{h.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
        ax.set_xticks(range(h.shape[1]), h.columns, rotation=30, ha="right"); ax.set_yticks(range(len(h)), h.index)
        ax.set_title("O que cada feature traz de NOVO em relação a P (AUROC além de P)"); fig.colorbar(im, ax=ax, shrink=0.6)
        _salvar(fig, pasta, "07_diagnostico_features.png", figs)

    es = R["estabilidade"]
    if len(es):
        s = es.groupby("experimento")[["Wang-Mendel (Jaccard das regras)", "Fuzzy aditivo (correlação dos pesos)"]].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(7, 3.5)); x = np.arange(len(s))
        for k, (col, cor) in enumerate([("Wang-Mendel (Jaccard das regras)", "#2a78d6"), ("Fuzzy aditivo (correlação dos pesos)", "#eb6834")]):
            ax.bar(x + (k - 0.5) * 0.38, s[(col, "mean")], 0.36, yerr=s[(col, "std")].fillna(0), color=cor, label=col, capsize=2)
        ax.set_xticks(x, s.index, rotation=30, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("estabilidade (1 = sempre igual)")
        ax.set_title("As regras se repetem quando os dados mudam?"); ax.legend(frameon=False, fontsize=8)
        _salvar(fig, pasta, "09_estabilidade_regras.png", figs)

    dx = R["dx_type"]
    if len(dx):
        s = dx.groupby("confirmação do diagnóstico")[["taxa de erro", "dúvida média (Wang-Mendel)"]].mean()
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        for a, col in zip(axs, s.columns):
            a.bar(s.index, s[col], color=["#2a78d6" if i == "histo" else "#b4b2a9" for i in s.index])
            a.set_title(col); a.tick_params(axis="x", rotation=30)
        fig.suptitle("HAM10000: erros e dúvida por forma de confirmação (histo = biópsia)")
        _salvar(fig, pasta, "11_ham_dx_type.png", figs)

    mf = R["multifotos"]
    if len(mf) and mf["AUROC (prever a mudança)"].notna().any():
        s = mf.groupby("método")["AUROC (prever a mudança)"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1.2))
        ax.barh(s.index, s.values, color=[NOSSO.get(m, "#b4b2a9") for m in s.index]); ax.axvline(0.5, color="#555", ls="--", lw=1)
        ax.set_xlabel("AUROC"); ax.set_title("HAM10000: a dúvida prevê se a rede muda de resposta em outra foto da mesma lesão?")
        _salvar(fig, pasta, "12_ham_varias_fotos.png", figs)

    ch = R["cifar10h"]
    if len(ch):
        for exp in ch.experimento.unique():
            s = ch[ch.experimento == exp].groupby("método")["correlação com a dúvida humana (Spearman)"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1.2))
            ax.barh(s.index, s.values, color=[NOSSO.get(m, "#b4b2a9") for m in s.index])
            ax.set_xlabel("correlação de Spearman com a dúvida de ~50 pessoas (CIFAR-10H)")
            ax.set_title(f"{exp}: quando a rede tem dúvida, as pessoas também têm?")
            _salvar(fig, pasta, f"13_cifar10h_{exp}.png", figs)
    return figs


# ----------------------------------------------------------------------------- HTML
TEXTOS = {
    "00": "Visão geral: cada célula é o AUROC médio (entre sementes) para detectar os erros da rede. Linhas NOSSO em negrito.",
    "01": "Todos os métodos num experimento. Azul/laranja = nossos; linha tracejada = só P (a referência a superar).",
    "02": "Encaminhando as X% mais incertas ao especialista, qual a acurácia nas que a rede decide sozinha? Curva mais alta = melhor.",
    "03": "Risco (taxa de erro) nas imagens que a rede decide sozinha, para cada cobertura. Curva mais baixa = melhor.",
    "04": "Curva ROC da detecção de erros (primeira semente).",
    "05": "Curva de opinião ao longo das camadas: nos erros, a rede concorda menos e mais tarde com a própria resposta.",
    "06": "Ganho de AUROC sobre só P com IC 95% em cada semente. Intervalo todo à direita do zero = ganho significativo.",
    "07": "Diagnóstico das features: AUROC dentro de faixas de P parecido. Acima de ~0,6 = traz informação nova.",
    "08": "Principais regras do Wang-Mendel (primeira semente).",
    "09": "Estabilidade das regras entre reamostragens.",
    "10": "Exemplos reais do HAM10000 com a curva de opinião.",
    "11": "HAM10000: lesões sem biópsia têm rótulo menos seguro — os erros e a dúvida se concentram nelas?",
    "12": "HAM10000: a dúvida medida numa foto prevê se a rede muda de resposta em outra foto da mesma lesão?",
    "13": "CIFAR-10H: correlação entre a dúvida de cada método e a dúvida de ~50 pessoas por imagem.",
}


def html(arq, tabelas, figs):
    css = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}"
           "table{border-collapse:collapse;font-size:12px;margin:0.5rem 0 1.5rem}td,th{border:1px solid #ddd;padding:3px 6px;text-align:right}"
           "th{background:#f3f2ee}td:first-child,th:first-child{text-align:left}img{max-width:100%;margin:0.3rem 0 1.2rem}"
           "h2{margin-top:2.2rem;border-bottom:1px solid #ddd}p.n{color:#555;font-size:14px}")
    corpo = ["<h1>Incerteza da rede lida por dentro + regras fuzzy</h1>",
             "<p class='n'>Todos os métodos usam a MESMA rede e as MESMAS imagens de teste. Escores: maior = mais chance de erro.</p>"]
    for t, df in tabelas:
        corpo += [f"<h2>{t}</h2>", df.to_html(float_format=lambda v: f"{v:.3f}", na_rep="—", index=False)]
    corpo.append("<h2>Gráficos</h2>")
    for f in figs:
        corpo += [f"<h3>{f}</h3><p class='n'>{TEXTOS.get(f[:2], '')}</p>", f"<img src='figuras/{f}'>"]
    arq.write_text(f"<!doctype html><meta charset='utf-8'><title>Relatório de incerteza</title><style>{css}</style>" + "\n".join(corpo), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--teste-rapido", action="store_true"); a = ap.parse_args()
    saida = C.SAIDA.parent / "resultados_teste_rapido" if a.teste_rapido else C.SAIDA
    R = ler_tudo(saida); ag = agregar(R["metricas"])
    pasta = saida / "figuras"; pasta.mkdir(parents=True, exist_ok=True)
    figs = graficos(R, ag, pasta, saida)
    geral = tabela_geral(ag).reset_index()
    pub = publicados(ag)
    nossos = ag[ag["método"].str.startswith("NOSSO") | (ag["método"] == "Só P")][
        ["experimento", "método", "AUROC (média)", "AUROC (desvio)", "E-AURC (x1000) (média)", "acurácia encaminhando 20% (média)",
         "ganho AUROC sobre só P (média)", "sementes com ganho significativo sobre só P", "sementes"]]
    redes = R["redes"]
    with pd.ExcelWriter(saida / "relatorio.xlsx") as xw:
        geral.to_excel(xw, sheet_name="visao_geral_AUROC", index=False)
        nossos.to_excel(xw, sheet_name="nossos_vs_so_P", index=False)
        ag.to_excel(xw, sheet_name="por_experimento_media_desvio", index=False)
        R["metricas"].to_excel(xw, sheet_name="por_semente", index=False)
        redes.to_excel(xw, sheet_name="redes", index=False)
        pub.to_excel(xw, sheet_name="numeros_publicados", index=False)
        R["estabilidade"].to_excel(xw, sheet_name="estabilidade_regras", index=False)
        R["diagnostico"].to_excel(xw, sheet_name="diagnostico_features", index=False)
        R["regras_wm"].to_excel(xw, sheet_name="regras_wang_mendel", index=False)
        R["regras_ad"].to_excel(xw, sheet_name="regras_fuzzy_aditivo", index=False)
        for k in ("dx_type", "multifotos", "cifar10h"):
            if len(R[k]):
                R[k].to_excel(xw, sheet_name=f"extra_{k}", index=False)
    tabs = [("Visão geral: AUROC (média ± desvio entre sementes)", geral), ("Nossos métodos × só P", nossos),
            ("Comparação com os números publicados (ganho sobre o softmax de cada um)", pub)]
    if len(redes):
        tabs.append(("Desempenho das redes", redes.drop(columns=[c for c in ("limiar",) if c in redes and redes[c].isna().all()])))
    html(saida / "relatorio.html", tabs, figs)
    print(f"Relatório: {saida / 'relatorio.html'}\nPlanilha:  {saida / 'relatorio.xlsx'}\nFiguras:   {pasta} ({len(figs)} arquivos)")
