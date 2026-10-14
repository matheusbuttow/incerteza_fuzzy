"""
extras.py — três validações com dados REAIS de incerteza (nada sintético):

1) HAM10000 · dx_type: o diagnóstico foi confirmado por biópsia (histo) ou não (consensus, follow_up, confocal)?
   Os erros e a dúvida se concentram nas lesões SEM biópsia (rótulo menos seguro)?
2) HAM10000 · várias fotos da mesma lesão: as outras fotos das lesões de teste (nunca usadas) passam pela rede.
   A dúvida medida na foto principal prevê se a rede muda de resposta em outra foto da MESMA lesão?
3) CIFAR-10H (Peterson et al. 2019): ~50 pessoas classificaram cada imagem de teste do CIFAR-10.
   Quando a rede está em dúvida, as pessoas também ficam?
"""
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import config as C

URL_CIFAR10H = "https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy"


def _auc(y, s):
    return roc_auc_score(y, s) if 0 < np.mean(y) < 1 else np.nan


def ham_dx_type(tab, meta, oof):
    lin = []
    for g, m in [("todas", np.ones(len(tab), bool))] + [(t, (meta.dx_type == t).values) for t in sorted(meta.dx_type.unique())] + \
                [("sem biópsia (todas)", (meta.dx_type != "histo").values)]:
        y = tab.erro.values[m]
        lin.append({"confirmação do diagnóstico": g, "imagens": int(m.sum()), "erros": int(y.sum()), "taxa de erro": y.mean() if m.sum() else np.nan,
                    "dúvida média (Wang-Mendel)": oof["NOSSO: Wang-Mendel"][m].mean() if m.sum() else np.nan,
                    "AUROC Wang-Mendel": _auc(y, oof["NOSSO: Wang-Mendel"][m]), "AUROC só P": _auc(y, oof["Só P"][m])})
    return pd.DataFrame(lin)


def ham_multifotos(tab, meta, oof, extras_meta, dec_extras):
    """Para cada lesão de teste com outras fotos: a rede deu outra resposta em alguma delas?"""
    ext = extras_meta.assign(dec=dec_extras)
    principal = pd.Series(tab.decisao.values, index=meta.lesion_id.values)
    ext["muda"] = ext.dec.values != principal.loc[ext.lesion_id].values
    muda = ext.groupby("lesion_id").muda.any()
    m = meta.lesion_id.isin(muda.index).values
    y = muda.loc[meta.lesion_id[m]].values.astype(int)
    lin = [{"método": k, "lesões com outras fotos": int(m.sum()), "% em que a rede muda de resposta": 100 * y.mean(),
            "AUROC (prever a mudança)": _auc(y, s[m])} for k, s in oof.items()]
    return pd.DataFrame(lin).sort_values("AUROC (prever a mudança)", ascending=False)


def cifar10h(oof):
    caminho = C.DADOS / "cifar10h-probs.npy"
    if not caminho.exists():
        C.DADOS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_CIFAR10H, caminho)
    H = np.load(caminho)                                                        # (10000, 10), ordem do teste do CIFAR-10
    ent = -(np.clip(H, 1e-9, 1) * np.log(np.clip(H, 1e-9, 1))).sum(1) / np.log(10)
    ambigua = (H.max(1) < 0.9).astype(int)                                      # menos de 90% das pessoas concordam
    lin = [{"método": k, "correlação com a dúvida humana (Spearman)": spearmanr(s, ent).correlation,
            "AUROC (achar imagens ambíguas para humanos)": _auc(ambigua, s), "% imagens ambíguas": 100 * ambigua.mean()}
           for k, s in oof.items()]
    return pd.DataFrame(lin).sort_values("correlação com a dúvida humana (Spearman)", ascending=False), ent
