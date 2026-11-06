"""
comparacao.py — todos os métodos de incerteza, avaliados do MESMO jeito nas MESMAS imagens de teste.

Sem treino (um número pronto):
  Softmax (MSP)            Hendrycks & Gimpel 2017
  Entropia do softmax
  Distância ao limiar      (só binário)
  TTA (vistas)             variação de P entre vistas
  MC Dropout (desvio)      Gal & Ghahramani 2016
  MC Dropout (entropia)    Gal & Ghahramani 2016
  Mahalanobis              Lee et al. 2018
  Deep kNN                 Papernot & McDaniel 2018 (vizinhos que discordam)
Treinados em validação cruzada (mesmos folds para todos):
  Só P                     binário (HAM): regressão logística com P e P² (aprende onde P erra perto do limiar);
                           várias classes: o próprio softmax (MSP), que já é a melhor forma de usar só a saída
  Chen et al. 2019         probes em cada camada + GBM
  Font et al. 2025         vizinhos em cada camada + regressão logística
  Gorbett & Blanchard 2022 SVM linear na penúltima camada
  ConfidNet (Corbière 2019) rede pequena que prevê a confiança na classe verdadeira
  NOSSO: Wang-Mendel       regras fuzzy (P + 2 features internas)
  NOSSO: Fuzzy aditivo     regras fuzzy de 1 condição com pesos (P + 3 features internas)
  NOSSO: Fuzzy aditivo (sem vistas)  o mesmo, proibido de usar as vistas: UMA passagem pela rede.
  NOSSO: Fuzzy aditivo + fronteira   o mesmo, podendo escolher também as pistas NOVAS (fronteira entre classes).
                           Mostra quanto do ganho vem da leitura interna e quanto das vistas (TTA).
Fuzzy de outros autores (fuzzy_outros.py; adaptados à tarefa, ver PROTOCOLO):
  Barna et al. 2026        possibilidade de erro sobre o softmax (sem treino)
  Aghaeipoor et al. 2023   regras Wang-Mendel sobre atribuições da penúltima camada
  Denoeux 2023 (ENNreg)    protótipos evidenciais com números fuzzy (mesmas características que o nosso)
  Msaddi e Kumbasar 2026   fuzzy tipo 2 intervalar + predição conforme (mesmas características que o nosso)
Cada método tem uma FAMÍLIA (os tópicos da revisão) e a ENTRADA que usa, para comparações justas dentro de cada família.
Os nossos métodos também são comparados diretamente com Chen et al. 2019 (IC 95% pareado).
Todos os escores: MAIOR = MAIS INCERTO (mais chance de a rede errar).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config as C
import fuzzy as fz
import fuzzy_outros as fo
from leitura import NOSSAS, NOVAS

warnings.filterwarnings("ignore")

SEM_TREINO = {"Softmax (MSP)": "msp", "Entropia do softmax": "entropia_softmax", "Distância ao limiar": "prox_p",
              "TTA (vistas)": "p_vistas_std", "MC Dropout (desvio)": "mc_dropout_std", "MC Dropout (entropia)": "mc_entropia",
              "Mahalanobis": "mahalanobis", "Deep kNN": "vizinhos_discordam"}
NOSSOS = ["NOSSO: Wang-Mendel", "NOSSO: Fuzzy aditivo", "NOSSO: Fuzzy aditivo (sem vistas)", "NOSSO: Fuzzy aditivo + fronteira"]
FUZZY_OUTROS = ["Barna et al. 2026", "Aghaeipoor et al. 2023", "Denoeux 2023 (ENNreg)", "Msaddi e Kumbasar 2026"]

# família (tópico da revisão) e entrada de cada método — comparações justas são feitas dentro da mesma família/entrada
FAMILIA = {
    "Softmax (MSP)": "1. Só a saída", "Entropia do softmax": "1. Só a saída", "Distância ao limiar": "1. Só a saída",
    "Só P": "1. Só a saída",
    "TTA (vistas)": "2. Várias passagens", "MC Dropout (desvio)": "2. Várias passagens", "MC Dropout (entropia)": "2. Várias passagens",
    "Mahalanobis": "3. Distância no espaço interno", "Deep kNN": "3. Distância no espaço interno",
    "Chen et al. 2019": "4. Camadas internas", "Font et al. 2025": "4. Camadas internas",
    "Gorbett & Blanchard 2022": "4. Camadas internas",
    "ConfidNet": "5. Confiança aprendida",
    "Barna et al. 2026": "8. Fuzzy (outros autores)", "Aghaeipoor et al. 2023": "8. Fuzzy (outros autores)",
    "Denoeux 2023 (ENNreg)": "8. Fuzzy (outros autores)", "Msaddi e Kumbasar 2026": "8. Fuzzy (outros autores)",
    "NOSSO: Fuzzy aditivo": "Nosso (4 + 8)", "NOSSO: Fuzzy aditivo (sem vistas)": "Nosso (4 + 8)",
    "NOSSO: Fuzzy aditivo + fronteira": "Nosso (4 + 8)",
    "NOSSO: Wang-Mendel": "Nosso — interno (fora do artigo)"}
ENTRADA = {
    "Softmax (MSP)": "saída", "Entropia do softmax": "saída", "Distância ao limiar": "saída", "Só P": "saída",
    "TTA (vistas)": "saída de várias vistas", "MC Dropout (desvio)": "saída de várias passagens", "MC Dropout (entropia)": "saída de várias passagens",
    "Mahalanobis": "penúltima camada", "Deep kNN": "penúltima camada",
    "Chen et al. 2019": "todas as camadas", "Font et al. 2025": "todas as camadas", "Gorbett & Blanchard 2022": "penúltima camada + saída",
    "ConfidNet": "penúltima camada",
    "Barna et al. 2026": "saída", "Aghaeipoor et al. 2023": "atribuições da penúltima camada",
    "Denoeux 2023 (ENNreg)": "nossas características", "Msaddi e Kumbasar 2026": "nossas características",
    "NOSSO: Fuzzy aditivo": "nossas características", "NOSSO: Fuzzy aditivo (sem vistas)": "nossas características (1 passagem)",
    "NOSSO: Fuzzy aditivo + fronteira": "nossas características + fronteira entre classes",
    "NOSSO: Wang-Mendel": "nossas características"}
VISTAS = ["vistas_mudam", "p_vistas_std", "trajetoria_vistas"]     # exigem passar a imagem várias vezes pela rede
REFERENCIA = {"Softmax (MSP)": "Hendrycks & Gimpel 2017", "Entropia do softmax": "—", "Distância ao limiar": "—",
              "TTA (vistas)": "—", "MC Dropout (desvio)": "Gal & Ghahramani 2016", "MC Dropout (entropia)": "Gal & Ghahramani 2016",
              "Mahalanobis": "Lee et al. 2018", "Deep kNN": "Papernot & McDaniel 2018", "Só P": "referência a superar",
              "Chen et al. 2019": "AISTATS 2019", "Font et al. 2025": "arXiv 2025", "Gorbett & Blanchard 2022": "WACV-W 2022",
              "ConfidNet": "Corbière et al., NeurIPS 2019", "NOSSO: Wang-Mendel": "proposto", "NOSSO: Fuzzy aditivo": "proposto",
              "NOSSO: Fuzzy aditivo (sem vistas)": "proposto, uma passagem",
              "NOSSO: Fuzzy aditivo + fronteira": "proposto, variante em teste",
              "Barna et al. 2026": "Applied Sciences 2026", "Aghaeipoor et al. 2023": "IEEE TFS 2023",
              "Denoeux 2023 (ENNreg)": "IEEE TFS 2023", "Msaddi e Kumbasar 2026": "IEEE TETCI 2026"}


def _p2(tab):
    p = tab[fz.RESPOSTA].values
    return np.c_[p, p ** 2]


def avaliar(tab, extras, binario, limiar, y_teste, semente):
    """Devolve (escores fora-do-fold por método, features escolhidas por fold, centros fixos)."""
    y = tab.erro.values
    fixos = {"p": [0.0, limiar, 1.0]} if binario and limiar is not None else None
    oof = {m: tab[c].values.astype(float) for m, c in SEM_TREINO.items() if not (m == "Distância ao limiar" and not binario)}
    treinados = (["Só P"] if binario else []) + ["Chen et al. 2019", "Font et al. 2025", "Gorbett & Blanchard 2022", "ConfidNet"] + NOSSOS \
        + [m for m in FUZZY_OUTROS if m != "Barna et al. 2026"]
    if not binario:
        oof["Só P"] = tab["msp"].values.astype(float)                       # várias classes: o próprio softmax
    for m in treinados:
        oof[m] = np.zeros(len(y))
    probs, pen = extras["probs"], extras["penultima"]
    top = np.sort(probs, 1)[:, ::-1][:, :min(3, probs.shape[1])]
    X = {"Só P": _p2(tab), "Chen et al. 2019": np.c_[extras["chen"], tab.p.values],
         "Font et al. 2025": np.c_[_p2(tab), tab[["font_dc", "font_lu"]].values],
         "Gorbett & Blanchard 2022": np.c_[pen, top, _p2(tab)]}
    tcp = probs[np.arange(len(y)), y_teste]                                   # confiança na classe verdadeira (ConfidNet)
    dec = tab["decisao"].values.astype(int)
    oof["Barna et al. 2026"] = fo.barna2026(probs, dec)
    atrib = fo.resumo_atribuicoes(extras["atrib"]) if "atrib" in extras else None
    if atrib is None:
        treinados.remove("Aghaeipoor et al. 2023"); del oof["Aghaeipoor et al. 2023"]
    escolhas = {m: [] for m in NOSSOS}
    cand = [c for c in NOSSAS if c in tab]
    cand_sem_vistas = [c for c in cand if c not in VISTAS]
    cand_fronteira = cand + [c for c in NOVAS if c in tab]
    if len(cand_fronteira) == len(cand):                                   # rodada sem as pistas novas
        treinados.remove("NOSSO: Fuzzy aditivo + fronteira"); del oof["NOSSO: Fuzzy aditivo + fronteira"]
        del escolhas["NOSSO: Fuzzy aditivo + fronteira"]
    for tr, te in StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=semente).split(tab, y):
        lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        for m, mod in ([("Só P", lr())] if binario else []) + [("Font et al. 2025", lr()),
                       ("Chen et al. 2019", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=semente)),
                       ("Gorbett & Blanchard 2022", make_pipeline(StandardScaler(), LinearSVC(C=0.01, class_weight="balanced", max_iter=20000)))]:
            mod.fit(X[m][tr], y[tr])
            oof[m][te] = mod.predict_proba(X[m][te])[:, 1] if hasattr(mod, "predict_proba") else mod.decision_function(X[m][te])
        cn = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=300, random_state=semente))
        oof["ConfidNet"][te] = -cn.fit(pen[tr], tcp[tr]).predict(pen[te])
        d = tab.iloc[tr]
        for m, k, Mod, cs in [("NOSSO: Wang-Mendel", C.FEATURES_WM, fz.WangMendel, cand),
                              ("NOSSO: Fuzzy aditivo", C.FEATURES_ADITIVO, fz.FuzzyAditivo, cand),
                              ("NOSSO: Fuzzy aditivo (sem vistas)", C.FEATURES_ADITIVO, fz.FuzzyAditivo, cand_sem_vistas),
                              ("NOSSO: Fuzzy aditivo + fronteira", C.FEATURES_ADITIVO, fz.FuzzyAditivo, cand_fronteira)]:
            if m not in escolhas:
                continue
            feats = fz.selecionar(d, cs, k); escolhas[m].append(feats)
            oof[m][te] = Mod(fixos).fit(d[feats].values, y[tr], feats).predict_proba(tab.iloc[te][feats].values)[:, 1]
            if m == "NOSSO: Fuzzy aditivo":
                f_ad = feats
        # fuzzy de outros autores: ENNreg e Msaddi recebem as MESMAS características que o nosso fuzzy aditivo
        Xf = tab[f_ad].values.astype(float)
        oof["Denoeux 2023 (ENNreg)"][te] = fo.ENNreg(semente=semente).fit(Xf[tr], tcp[tr]).score(Xf[te])
        oof["Msaddi e Kumbasar 2026"][te] = fo.Msaddi2026(semente=semente).fit(Xf[tr], tcp[tr]).score(Xf[te])
        if atrib is not None:
            oof["Aghaeipoor et al. 2023"][te] = fo.Aghaeipoor2023().fit(atrib[tr], y[tr]).score(atrib[te])
    return oof, escolhas, fixos


# ----------------------------------------------------------------------------- três zonas (tipo 1)
def _seguro(ajusta):
    """Se os dados de ajuste têm uma classe só (rede que quase nunca erra ou quase sempre erra), devolve uma nota constante."""
    return lambda X, y: ajusta(X, y) if len(np.unique(y)) > 1 else (lambda Z, c=float(np.mean(y)): np.full(len(Z), c))


def _oof_interno(ajusta, X, y, semente, k=3):
    """Notas fora-da-dobra DENTRO do treino (para escolher os limiares sem olhar o teste nem notas de treino)."""
    s = np.zeros(len(y)); k = min(k, int(min(np.bincount(y.astype(int), minlength=2))))
    if k < 2:
        return np.full(len(y), float(np.mean(y)))
    for a, b in StratifiedKFold(k, shuffle=True, random_state=semente + 1).split(X, y):
        s[b] = _seguro(ajusta)(X[a], y[a])(X[b])
    return s


def zonas(tab, binario, limiar, semente):
    """Classifica cada resposta da rede em SEGURO / INCERTO / ALARME (fuzzy aditivo e Só P, mesmas dobras de `avaliar`).
    Limiares escolhidos só no treino de cada dobra: SEGURO = erro ≤ ZONA_META_SEGURO_REL × erro da rede;
    ALARME = erro ≥ ZONA_META_ALARME (com garantia de 95% se ZONA_GARANTIA). Devolve (zona por imagem, resumo)."""
    y = tab.erro.values; n = len(y)
    fixos = {"p": [0.0, limiar, 1.0]} if binario and limiar is not None else None
    cand = [c for c in NOSSAS if c in tab]
    z = {m: np.full(n, "", dtype=object) for m in ("NOSSO: Fuzzy aditivo", "Só P")}
    metas = []
    for tr, te in StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=semente).split(tab, y):
        d = tab.iloc[tr]; ytr = y[tr]                                         # seleção só no treino da dobra
        meta_seg = C.ZONA_META_SEGURO_REL * ytr.mean(); metas.append(meta_seg)
        feats = fz.selecionar(d, cand, C.FEATURES_ADITIVO)
        fz_fit = lambda X, yy, f=feats: (lambda m: (lambda Z: m.predict_proba(Z)[:, 1]))(fz.FuzzyAditivo(fixos).fit(X, yy, f))
        if binario:
            sp_fit = lambda X, yy: (lambda m: (lambda Z: m.predict_proba(Z)[:, 1]))(
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X, yy))
            Xsp = _p2(tab)
        else:
            sp_fit = lambda X, yy: (lambda Z: Z[:, 0]); Xsp = tab[["msp"]].values
        for m, ajusta, X in [("NOSSO: Fuzzy aditivo", fz_fit, tab[feats].values), ("Só P", sp_fit, Xsp)]:
            s_in = _oof_interno(ajusta, X[tr], ytr, semente)
            t_seg = fz.limiar_seguro(s_in, ytr, meta_seg, C.ZONA_GARANTIA)
            t_al = fz.limiar_alarme(s_in, ytr, C.ZONA_META_ALARME, C.ZONA_GARANTIA)
            z[m][te] = fz.zona(_seguro(ajusta)(X[tr], ytr)(X[te]), t_seg, t_al)
    lin = []
    for m, zz in z.items():
        for nome_z in ("alarme", "incerto", "seguro"):
            mz = zz == nome_z
            lin.append({"método": m, "zona": nome_z, "% das imagens": mz.mean(), "imagens": int(mz.sum()),
                        "taxa de erro da rede na zona": y[mz].mean() if mz.any() else np.nan,
                        "erros na zona": int(y[mz].sum()), "% dos erros": y[mz].sum() / max(y.sum(), 1)})
        prev, yb = zz != "seguro", y.astype(bool)
        lin.append({"método": m, "zona": "revisar (alarme + incerto)", "% das imagens": prev.mean(), "imagens": int(prev.sum()),
                    "taxa de erro da rede na zona": y[prev].mean() if prev.any() else np.nan,
                    "erros na zona": int(y[prev].sum()), "% dos erros": y[prev].sum() / max(y.sum(), 1)})
    res = pd.DataFrame(lin); res["meta seguro (média das dobras)"] = np.mean(metas); res["erro da rede"] = y.mean()
    return pd.DataFrame({k.replace("NOSSO: ", "zona "): v for k, v in z.items()}), res


# ----------------------------------------------------------------------------- métricas
def aurc(erro, s):
    e = np.asarray(erro, float)[np.argsort(s, kind="stable")]
    risco = np.cumsum(e) / np.arange(1, len(e) + 1); r = e.mean()
    otimo = r + (1 - r) * np.log(1 - r) if 0 < r < 1 else 0.0
    return 1000 * risco.mean(), 1000 * (risco.mean() - otimo)


def metricas(y, s):
    """y: 1 = a rede errou. s: maior = mais incerto."""
    a, ea = aurc(y, s); o = np.argsort(s, kind="stable")
    ac = -s[y == 0]; thr = np.quantile(ac, 0.05)                                # aceita 95% dos acertos
    out = {"AUROC": roc_auc_score(y, s), "AUPR-erro": average_precision_score(y, s),
           "AUPR-acerto": average_precision_score(1 - y, -s), "FPR@95%TPR": float(np.mean(-s[y == 1] >= thr)),
           "AURC (x1000)": a, "E-AURC (x1000)": ea}
    for fr in C.ENCAMINHAR:
        fica = o[: max(1, int(round((1 - fr) * len(o))))]
        out[f"acurácia encaminhando {int(round(100 * fr))}%"] = 1 - y[fica].mean()
    return out


def tabela(y, oof, base="Só P", rival="Chen et al. 2019", semente=0):
    """Métricas + ganho de AUROC sobre só P e sobre Chen et al. 2019, com IC 95% (bootstrap pareado:
    as MESMAS reamostragens para todos os métodos)."""
    r = np.random.default_rng(semente)
    boot = [i for i in (r.integers(0, len(y), len(y)) for _ in range(C.N_BOOT)) if y[i].min() != y[i].max()]
    auc_b = {m: np.array([roc_auc_score(y[i], np.asarray(s, float)[i]) for i in boot]) for m, s in oof.items()}
    lin = []
    for m, s in oof.items():
        s = np.asarray(s, float); mt = metricas(y, s)
        d, dc = auc_b[m] - auc_b[base], auc_b[m] - auc_b[rival]
        lin.append({"método": m, "família": FAMILIA.get(m, ""), "entrada": ENTRADA.get(m, ""), "referência": REFERENCIA.get(m, ""), **mt,
                    "ganho AUROC sobre só P": mt["AUROC"] - roc_auc_score(y, oof[base]),
                    "IC95 ganho inf": np.percentile(d, 2.5), "IC95 ganho sup": np.percentile(d, 97.5),
                    "ganho AUROC sobre Chen": mt["AUROC"] - roc_auc_score(y, oof[rival]),
                    "IC95 Chen inf": np.percentile(dc, 2.5), "IC95 Chen sup": np.percentile(dc, 97.5)})
    return pd.DataFrame(lin)


def curvas(y, oof):
    """Curva de rejeição, risco-cobertura e ROC de cada método (formato longo, para os gráficos)."""
    rej, rc, roc = [], [], []
    cob = np.linspace(0.02, 1, 50)
    for m, s in oof.items():
        o = np.argsort(s, kind="stable")
        for fr in np.linspace(0, 0.5, 21):
            fica = o[: max(1, int(round((1 - fr) * len(o))))]
            rej.append({"método": m, "% encaminhado": 100 * fr, "acurácia nas que ficam": 1 - y[fica].mean()})
        for c in cob:
            fica = o[: max(1, int(round(c * len(o))))]
            rc.append({"método": m, "cobertura": c, "risco": y[fica].mean()})
        fpr, tpr, _ = roc_curve(y, s)
        j = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
        roc += [{"método": m, "FPR": a, "TPR": b} for a, b in zip(fpr[j], tpr[j])]
    return pd.DataFrame(rej), pd.DataFrame(rc), pd.DataFrame(roc)


def regras_finais(tab, fixos):
    """Ajusta os dois sistemas fuzzy em todo o teste para mostrar as regras e medir a estabilidade."""
    cand = [c for c in NOSSAS if c in tab]
    f_wm, f_ad = fz.selecionar(tab, cand, C.FEATURES_WM), fz.selecionar(tab, cand, C.FEATURES_ADITIVO)
    wm = fz.WangMendel(fixos).fit(tab[f_wm].values, tab.erro.values, f_wm)
    ad = fz.FuzzyAditivo(fixos).fit(tab[f_ad].values, tab.erro.values, f_ad)
    est = {"Wang-Mendel (Jaccard das regras)": fz.estabilidade_wm(tab, f_wm, fixos),
           "Fuzzy aditivo (correlação dos pesos)": fz.estabilidade_aditivo(tab, f_ad, fixos),
           "features Wang-Mendel": ", ".join(f_wm), "features fuzzy aditivo": ", ".join(f_ad), "nº regras Wang-Mendel": len(wm.regras_)}
    return wm.regras(), ad.regras(), est
