"""
tabelas_latex.py — gera as tabelas do artigo em LaTeX (para o Overleaf) a partir do relatorio.xlsx.

  python tabelas_latex.py                                            # usa resultados/relatorio.xlsx
  python tabelas_latex.py historico/teste1_2026-10-16/relatorio.xlsx  # usa o teste congelado

Saída (mesma pasta da planilha): tabelas_pt.tex e tabelas_en.tex.
No Overleaf: \\usepackage{booktabs,multirow,graphicx} no preâmbulo e \\input{tabelas_en} no texto.
Tabelas: (1) AUROC de todos os métodos; (2) erros encontrados revisando 10/20/30%; (3) ganho sobre o softmax
por cenário; (4) significância contra Chen et al. 2019 e Só P; (5) redes usadas.
"""
import sys
from pathlib import Path

import pandas as pd

ORDEM = ["ham10000_todas", "ham10000", "cifar10_ruido30", "cifar100_vgg16", "cifar10", "cifar10_vgg16", "svhn", "mnist"]
CENARIO = {"pt": {"ham10000_todas": "HAM10000 (todas)", "ham10000": "HAM10000 (3000)", "cifar10_ruido30": "CIFAR-10 ruído 30\\%",
                  "cifar100_vgg16": "CIFAR-100", "cifar10": "CIFAR-10 (ResNet)", "cifar10_vgg16": "CIFAR-10 (VGG)",
                  "svhn": "SVHN", "mnist": "MNIST"},
           "en": {"ham10000_todas": "HAM10000 (all)", "ham10000": "HAM10000 (3000)", "cifar10_ruido30": "CIFAR-10 30\\% noise",
                  "cifar100_vgg16": "CIFAR-100", "cifar10": "CIFAR-10 (ResNet)", "cifar10_vgg16": "CIFAR-10 (VGG)",
                  "svhn": "SVHN", "mnist": "MNIST"}}
CLASSES = {"ham10000_todas": 2, "ham10000": 2, "cifar10_ruido30": 10, "cifar100_vgg16": 100, "cifar10": 10,
           "cifar10_vgg16": 10, "svhn": 10, "mnist": 10}
ARQ = {"ham10000_todas": "EfficientNet-B0", "ham10000": "EfficientNet-B0", "cifar10_ruido30": "ResNet-32",
       "cifar100_vgg16": "VGG-16", "cifar10": "ResNet-32", "cifar10_vgg16": "VGG-16", "svhn": "CNN", "mnist": "MLP"}
METODOS = [  # (nome na planilha, pt, en, grupo)
    ("Softmax (MSP)", "Softmax (MSP)", "Softmax (MSP)", 0),
    ("Entropia do softmax", "Entropia do softmax", "Softmax entropy", 0),
    ("Distância ao limiar", "Distância ao limiar", "Distance to threshold", 0),
    ("TTA (vistas)", "TTA", "TTA", 0),
    ("MC Dropout (entropia)", "MC Dropout", "MC Dropout", 0),
    ("Mahalanobis", "Mahalanobis", "Mahalanobis", 0),
    ("Deep kNN", "Deep kNN", "Deep kNN", 0),
    ("Só P", "Só P$^\\dagger$", "Output only$^\\dagger$", 1),
    ("Chen et al. 2019", "Chen et al.~\\cite{chen2019}", "Chen et al.~\\cite{chen2019}", 1),
    ("Font et al. 2025", "Font et al.~\\cite{font2025}", "Font et al.~\\cite{font2025}", 1),
    ("Gorbett & Blanchard 2022", "Gorbett e Blanchard~\\cite{gorbett2022}", "Gorbett and Blanchard~\\cite{gorbett2022}", 1),
    ("ConfidNet", "ConfidNet~\\cite{corbiere2019}", "ConfidNet~\\cite{corbiere2019}", 1),
    ("NOSSO: Wang-Mendel", "Wang-Mendel (nosso)", "Wang-Mendel (ours)", 2),
    ("NOSSO: Fuzzy aditivo (sem vistas)", "Fuzzy aditivo, sem vistas (nosso)", "Additive fuzzy, no views (ours)", 2),
    ("NOSSO: Fuzzy aditivo", "\\textbf{Fuzzy aditivo (nosso)}", "\\textbf{Additive fuzzy (ours)}", 2)]
CURTOS = ["Softmax (MSP)", "MC Dropout (entropia)", "TTA (vistas)", "Chen et al. 2019", "Gorbett & Blanchard 2022", "Só P",
          "NOSSO: Fuzzy aditivo (sem vistas)", "NOSSO: Fuzzy aditivo"]
T = {"pt": dict(
        cap1="AUROC para detectar os erros da rede (média $\\pm$ desvio entre sementes). Em negrito, a maior média de cada cenário.",
        nota1="$^\\dagger$ Binário: regressão logística em $P$ e $P^2$; várias classes: idêntico ao softmax.",
        metodo="Método", cap2="Erros da rede encontrados ao encaminhar ao especialista 10\\%, 20\\% e 30\\% das imagens mais incertas "
        "(a cada 1.000 imagens; MNIST a cada 10.000). Entre parênteses, a fração de todos os erros da rede.",
        acaso="Ao acaso", erros="erros", rev="Revisando", cap3="Ganho de AUROC do fuzzy aditivo sobre o softmax por cenário.",
        cen="Cenário", cl="Classes", taxa="Erro da rede", ganho="Ganho", cap4="Rodadas (sementes) em que o fuzzy aditivo ganha "
        "com significância (IC 95\\%, bootstrap pareado) de Chen et al. e de Só P.", gc="Ganha de Chen", pc="Perde para Chen",
        gp="Ganha de Só P", total="Total", cap5="Redes avaliadas (congeladas). Média entre sementes.", arq="Rede",
        teste="Imagens de teste", acc="Acurácia", sem="Sementes"),
     "en": dict(
        cap1="AUROC for detecting the network's errors (mean $\\pm$ std over seeds). Best mean per scenario in bold.",
        nota1="$^\\dagger$ Binary: logistic regression on $P$ and $P^2$; multiclass: identical to softmax.",
        metodo="Method", cap2="Network errors found when the 10\\%, 20\\% and 30\\% most uncertain images are referred to an expert "
        "(per 1,000 images; MNIST per 10,000). In parentheses, the fraction of all network errors.",
        acaso="Random", erros="errors", rev="Reviewing", cap3="AUROC gain of the additive fuzzy model over softmax per scenario.",
        cen="Scenario", cl="Classes", taxa="Error rate", ganho="Gain", cap4="Runs (seeds) in which the additive fuzzy model is "
        "significantly better (95\\% CI, paired bootstrap) than Chen et al. and the output-only baseline.", gc="Beats Chen",
        pc="Loses to Chen", gp="Beats output only", total="Total", cap5="Evaluated (frozen) networks. Mean over seeds.",
        arq="Network", teste="Test images", acc="Accuracy", sem="Seeds")}


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def sinal(x, nd=3):
    return f"{x:+.{nd}f}".replace("-", "$-$")


def duas(txt):
    """Cabeçalho em duas linhas sem pacote extra: 'CIFAR-10 (VGG)' -> CIFAR-10 / (VGG)."""
    a, sep, b = txt.partition(" (") if " (" in txt else txt.rpartition(" ")
    if not sep:
        return txt
    b = ("(" + b) if sep == " (" else b
    return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{a}\\\\{b}\\end{{tabular}}"


def ambiente(tipo, cap, label, largura, colunas, cab, corpo, nota=""):
    linhas = "\n".join("  " + l for l in corpo)
    return (f"\\begin{{{tipo}}}[t]\n\\centering\n\\caption{{{cap}}}\n\\label{{{label}}}\n"
            f"\\resizebox{{{largura}}}{{!}}{{%\n\\begin{{tabular}}{{{colunas}}}\n  \\toprule\n  {cab} \\\\\n  \\midrule\n"
            f"{linhas}\n  \\bottomrule\n\\end{{tabular}}}}\n" + (f"\n\\vspace{{2pt}}\n{{\\footnotesize {nota}}}\n" if nota else "")
            + f"\\end{{{tipo}}}\n")


def nomes(L):
    return {m: (pt if L == "pt" else en) for m, pt, en, _ in METODOS}


def t_auroc(md, L, exps):
    t, nm = T[L], nomes(L)
    piv_m = md.pivot(index="método", columns="experimento", values="AUROC (média)")
    piv_s = md.pivot(index="método", columns="experimento", values="AUROC (desvio)")
    melhor = piv_m[exps].max().round(3)
    corpo, grupo = [], None
    for m, *_, g in METODOS:
        if m not in piv_m.index:
            continue
        if grupo is not None and g != grupo:
            corpo.append("\\midrule")
        grupo = g
        cel = []
        for e in exps:
            v, s_ = piv_m.loc[m, e], piv_s.loc[m, e]
            if pd.isna(v):
                cel.append("--"); continue
            num = f"\\textbf{{{fmt(v)}}}" if round(v, 3) == melhor[e] else fmt(v)
            cel.append(f"{num}\\,{{\\scriptsize$\\pm${fmt(s_)}}}")
        corpo.append(nm[m] + " & " + " & ".join(cel) + " \\\\")
    cab = t["metodo"] + " & " + " & ".join(duas(CENARIO[L][e]) for e in exps)
    return ambiente("table*", t["cap1"], "tab:auroc", "\\textwidth", "l" + "c" * len(exps), cab, corpo, t["nota1"])


def encontrados(ps, rd):
    rd = rd.assign(taxa=rd["erros da rede"] / rd["imagens de teste"])
    d = ps.merge(rd[["experimento", "semente", "taxa"]], on=["experimento", "semente"])
    for fr in (10, 20, 30):
        d[f"c{fr}"] = d.taxa - (1 - d[f"acurácia encaminhando {fr}%"]) * (1 - fr / 100)
    return d.groupby(["experimento", "método"])[["c10", "c20", "c30", "taxa"]].mean()


def t_encontrados(ps, rd, L, exps):
    t, nm = T[L], nomes(L)
    enc = encontrados(ps, rd)

    def bloco(e, primeiro):
        g = enc.loc[e]; esc = 10000 if e == "mnist" else 1000; E = g.taxa.mean() * esc
        binario = CLASSES[e] == 2; c = [] if primeiro else ["\\midrule"]
        por = (" (" + ("a cada 10.000" if L == "pt" else "per 10,000") + ")") if esc == 10000 else ""
        c.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{CENARIO[L][e]}}} --- {E:.0f} {t['erros']}{por}}} \\\\")
        c.append(t["acaso"] + " & " + " & ".join(f"{E * f / 100:.0f} ({f}\\%)" for f in (10, 20, 30)) + " \\\\")
        ms = [m for m in CURTOS if m in g.index and (binario or m != "Só P")]
        vals = {m: [g.loc[m, f"c{f}"] * esc for f in (10, 20, 30)] for m in ms}
        top = [max(round(v[k]) for v in vals.values()) for k in range(3)]
        for m in ms:
            cel = [f"{round(v):d} ({100 * v / E:.0f}\\%)" for v in vals[m]]
            cel = [f"\\textbf{{{x}}}" if round(vals[m][k]) == top[k] else x for k, x in enumerate(cel)]
            nome = nm[m].replace("$^\\dagger$", "")
            if m == "Softmax (MSP)" and not binario:
                nome += " = " + nm["Só P"].replace("$^\\dagger$", "")
            c.append(nome + " & " + " & ".join(cel) + " \\\\")
        return c

    cab = t["metodo"] + " & " + " & ".join(duas(f"{t['rev']} {f}\\%") for f in (10, 20, 30))
    meio = (len(exps) + 1) // 2
    lados = []
    for parte in (exps[:meio], exps[meio:]):
        corpo = [l for i, e in enumerate(parte) for l in bloco(e, i == 0)]
        lados.append("\\begin{minipage}[t]{0.495\\textwidth}\n\\centering\n\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
                     "\\begin{tabular}[t]{lccc}\n  \\toprule\n  " + cab + " \\\\\n  \\midrule\n"
                     + "\n".join("  " + l for l in corpo) + "\n  \\bottomrule\n\\end{tabular}\n\\end{minipage}")
    return ("\\begin{table*}[t]\n\\centering\n\\caption{" + t["cap2"] + "}\n\\label{tab:encontrados}\n"
            + lados[0] + "\\hfill\n" + lados[1] + "\n\\end{table*}\n")


def t_ganho(md, rd, L, exps):
    t = T[L]; piv = md.pivot(index="método", columns="experimento", values="AUROC (média)")
    taxa = (rd["erros da rede"] / rd["imagens de teste"]).groupby(rd.experimento).mean()
    g = {e: piv.loc["NOSSO: Fuzzy aditivo", e] - piv.loc["Softmax (MSP)", e] for e in exps}
    dec = "," if L == "pt" else "."
    corpo = [f"{CENARIO[L][e]} & {CLASSES[e]} & {100 * taxa[e]:.1f}\\% & {fmt(piv.loc['Softmax (MSP)', e])} & "
             f"{fmt(piv.loc['NOSSO: Fuzzy aditivo', e])} & {sinal(g[e])} \\\\" for e in sorted(exps, key=g.get)]
    ad = "Fuzzy aditivo" if L == "pt" else "Additive fuzzy"
    cab = f"{t['cen']} & {t['cl']} & {duas(t['taxa'])} & Softmax & {duas(ad)} & {t['ganho']}"
    return ambiente("table", t["cap3"], "tab:ganho", "\\columnwidth", "lccccc", cab, corpo)


def t_signif(nv, L, exps):
    t = T[L]; a = nv[nv["método"] == "NOSSO: Fuzzy aditivo"].set_index("experimento")
    cols = ["sementes com ganho significativo sobre Chen", "sementes com perda significativa para Chen",
            "sementes com ganho significativo sobre só P", "sementes"]
    corpo = [f"{CENARIO[L][e]} & " + " & ".join(str(int(a.loc[e, c])) for c in cols) + " \\\\" for e in exps]
    tot = [int(a.loc[exps, c].sum()) for c in cols]
    corpo += ["\\midrule", f"\\textbf{{{t['total']}}} & " + " & ".join(f"\\textbf{{{v}}}" for v in tot) + " \\\\"]
    cab = f"{t['cen']} & {duas(t['gc'])} & {duas(t['pc'])} & {duas(t['gp'])} & {t['sem']}"
    return ambiente("table", t["cap4"], "tab:signif", "\\columnwidth", "lcccc", cab, corpo)


def t_redes(rd, L, exps):
    t = T[L]; g = rd.groupby("experimento"); mil = "." if L == "pt" else ","
    corpo = [f"{CENARIO[L][e]} & {ARQ[e]} & {CLASSES[e]} & "
             + f"{g.get_group(e)['imagens de teste'].mean():,.0f}".replace(",", mil)
             + f" & {g.get_group(e)['acurácia teste'].mean():.3f} & {len(g.get_group(e))} \\\\" for e in exps]
    cab = f"{t['cen']} & {t['arq']} & {t['cl']} & {duas(t['teste'])} & {t['acc']} & {t['sem']}"
    return ambiente("table", t["cap5"], "tab:redes", "\\columnwidth", "llcccc", cab, corpo)


if __name__ == "__main__":
    xl = Path(sys.argv[1] if len(sys.argv) > 1 else "resultados/relatorio.xlsx")
    x = pd.read_excel(xl, sheet_name=None)
    md, ps, rd, nv = x["por_experimento_media_desvio"], x["por_semente"], x["redes"], x["nossos_vs_so_P"]
    exps = [e for e in ORDEM if e in set(md.experimento)]
    for L in ("pt", "en"):
        cab = (f"% Gerado por tabelas_latex.py a partir de {xl.name}.\n"
               "% Preâmbulo: \\usepackage{booktabs,multirow,graphicx}\n\n")
        partes = [t_redes(rd, L, exps), t_auroc(md, L, exps), t_encontrados(ps, rd, L, exps),
                  t_ganho(md, rd, L, exps), t_signif(nv, L, exps)]
        titulos = ["Redes", "AUROC", "Erros encontrados", "Ganho sobre o softmax", "Significância"]
        txt = cab + "\n".join(f"% {'=' * 66}\n% Tabela {i}: {n}\n% {'=' * 66}\n{p}" for i, (n, p) in
                               enumerate(zip(titulos, partes), 1))
        (xl.parent / f"tabelas_{L}.tex").write_text(txt, encoding="utf-8")
        print(f"{xl.parent / f'tabelas_{L}.tex'}")
