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


def t_auroc(md, L, exps):
    t, nm = T[L], {m: (pt if L == "pt" else en) for m, pt, en, _ in METODOS}
    piv_m = md.pivot(index="método", columns="experimento", values="AUROC (média)")
    piv_s = md.pivot(index="método", columns="experimento", values="AUROC (desvio)")
    melhor = piv_m[exps].max()
    lin, grupo = [], 0
    for m, *_, g in METODOS:
        if m not in piv_m.index:
            continue
        if g != grupo:
            lin.append("\\midrule"); grupo = g
        cel = []
        for e in exps:
            v, s = piv_m.loc[m, e], piv_s.loc[m, e]
            if pd.isna(v):
                cel.append("--"); continue
            c = f"{fmt(v)}$\\pm${fmt(s)}"
            cel.append(f"\\textbf{{{c}}}" if round(v, 3) == round(melhor[e], 3) else c)
        lin.append(nm[m] + " & " + " & ".join(cel) + " \\\\")
    cab = " & ".join(f"\\rotatebox{{0}}{{{CENARIO[L][e]}}}" for e in exps)
    return ("\\begin{table*}[t]\n\\centering\n\\caption{" + t["cap1"] + "}\n\\label{tab:auroc}\n\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{l" + "c" * len(exps) + "}\n\\toprule\n" + t["metodo"] + " & " + cab + " \\\\\n\\midrule\n"
            + "\n".join(lin[1:] if lin and lin[0] == "\\midrule" else lin) + "\n\\bottomrule\n\\end{tabular}}\n\n\\vspace{2pt}\n"
            "\\footnotesize " + t["nota1"] + "\n\\end{table*}\n")


def encontrados(ps, rd):
    rd = rd.assign(taxa=rd["erros da rede"] / rd["imagens de teste"])
    d = ps.merge(rd[["experimento", "semente", "taxa"]], on=["experimento", "semente"])
    for fr in (10, 20, 30):
        d[f"c{fr}"] = d.taxa - (1 - d[f"acurácia encaminhando {fr}%"]) * (1 - fr / 100)
    return d.groupby(["experimento", "método"])[["c10", "c20", "c30", "taxa"]].mean()


def t_encontrados(ps, rd, L, exps):
    t, nm = T[L], {m: (pt if L == "pt" else en) for m, pt, en, _ in METODOS}
    enc = encontrados(ps, rd); lin = []
    for e in exps:
        g = enc.loc[e]; esc = 10000 if e == "mnist" else 1000; E = g.taxa.mean() * esc
        binario = CLASSES[e] == 2
        lin.append(f"\\midrule\n\\multicolumn{{4}}{{l}}{{\\textit{{{CENARIO[L][e]}}} --- {E:.0f} {t['erros']}}} \\\\")
        lin.append(t["acaso"] + " & " + " & ".join(f"{E * f / 100:.0f} ({f}\\%)" for f in (10, 20, 30)) + " \\\\")
        ms = [m for m in CURTOS if m in g.index and (binario or m != "Só P")]
        vals = {m: [g.loc[m, f"c{f}"] * esc for f in (10, 20, 30)] for m in ms}
        top = [max(round(v[k]) for v in vals.values()) for k in range(3)]
        for m in ms:
            cel = [f"{round(v):d} ({100 * v / E:.0f}\\%)" for v in vals[m]]
            cel = [f"\\textbf{{{c}}}" if round(vals[m][k]) == top[k] else c for k, c in enumerate(cel)]
            nome = nm[m].replace("$^\\dagger$", "") if binario else nm[m]
            if m == "Softmax (MSP)" and not binario:
                nome += " = " + nm["Só P"].replace("$^\\dagger$", "")
            lin.append(nome + " & " + " & ".join(cel) + " \\\\")
    cab = " & ".join(f"{t['rev']} {f}\\%" for f in (10, 20, 30))
    return ("\\begin{table}[t]\n\\centering\n\\caption{" + t["cap2"] + "}\n\\label{tab:encontrados}\n\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{lccc}\n\\toprule\n" + t["metodo"] + " & " + cab + " \\\\\n" + "\n".join(lin)
            + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def t_ganho(md, rd, L, exps):
    t = T[L]; piv = md.pivot(index="método", columns="experimento", values="AUROC (média)")
    taxa = (rd["erros da rede"] / rd["imagens de teste"]).groupby(rd.experimento).mean()
    ordem = sorted(exps, key=lambda e: piv.loc["NOSSO: Fuzzy aditivo", e] - piv.loc["Softmax (MSP)", e])
    lin = [f"{CENARIO[L][e]} & {CLASSES[e]} & {100 * taxa[e]:.1f}\\% & {fmt(piv.loc['Softmax (MSP)', e])} & "
           f"{fmt(piv.loc['NOSSO: Fuzzy aditivo', e])} & {piv.loc['NOSSO: Fuzzy aditivo', e] - piv.loc['Softmax (MSP)', e]:+.3f} \\\\"
           for e in ordem]
    ad = "Fuzzy aditivo" if L == "pt" else "Additive fuzzy"
    return ("\\begin{table}[t]\n\\centering\n\\caption{" + t["cap3"] + "}\n\\label{tab:ganho}\n\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{lccccc}\n\\toprule\n" + f"{t['cen']} & {t['cl']} & {t['taxa']} & Softmax & {ad} & {t['ganho']}"
            " \\\\\n\\midrule\n" + "\n".join(lin) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def t_signif(nv, L, exps):
    t = T[L]; a = nv[nv["método"] == "NOSSO: Fuzzy aditivo"].set_index("experimento")
    cols = ["sementes com ganho significativo sobre Chen", "sementes com perda significativa para Chen",
            "sementes com ganho significativo sobre só P", "sementes"]
    lin = [f"{CENARIO[L][e]} & " + " & ".join(str(int(a.loc[e, c])) for c in cols) + " \\\\" for e in exps]
    tot = [int(a.loc[exps, c].sum()) for c in cols]
    return ("\\begin{table}[t]\n\\centering\n\\caption{" + t["cap4"] + "}\n\\label{tab:signif}\n\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n" + f"{t['cen']} & {t['gc']} & {t['pc']} & {t['gp']} & {t['sem']}"
            " \\\\\n\\midrule\n" + "\n".join(lin) + f"\n\\midrule\n{t['total']} & " + " & ".join(map(str, tot))
            + " \\\\\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def t_redes(rd, L, exps):
    t = T[L]; g = rd.groupby("experimento")
    lin = [f"{CENARIO[L][e]} & {ARQ[e]} & {CLASSES[e]} & {g.get_group(e)['imagens de teste'].mean():,.0f} & "
           f"{g.get_group(e)['acurácia teste'].mean():.3f} & {len(g.get_group(e))} \\\\".replace(",", "." if L == "pt" else ",")
           for e in exps]
    return ("\\begin{table}[t]\n\\centering\n\\caption{" + t["cap5"] + "}\n\\label{tab:redes}\n\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{llcccc}\n\\toprule\n" + f"{t['cen']} & {t['arq']} & {t['cl']} & {t['teste']} & {t['acc']} & {t['sem']}"
            " \\\\\n\\midrule\n" + "\n".join(lin) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


if __name__ == "__main__":
    xl = Path(sys.argv[1] if len(sys.argv) > 1 else "resultados/relatorio.xlsx")
    x = pd.read_excel(xl, sheet_name=None)
    md, ps, rd, nv = x["por_experimento_media_desvio"], x["por_semente"], x["redes"], x["nossos_vs_so_P"]
    exps = [e for e in ORDEM if e in set(md.experimento)]
    for L in ("pt", "en"):
        cab = (f"% Gerado por tabelas_latex.py a partir de {xl.name}.\n"
               "% Preâmbulo: \\usepackage{booktabs,multirow,graphicx}\n\n")
        txt = cab + "\n".join([t_redes(rd, L, exps), t_auroc(md, L, exps), t_encontrados(ps, rd, L, exps),
                               t_ganho(md, rd, L, exps), t_signif(nv, L, exps)])
        (xl.parent / f"tabelas_{L}.tex").write_text(txt, encoding="utf-8")
        print(f"{xl.parent / f'tabelas_{L}.tex'}")
