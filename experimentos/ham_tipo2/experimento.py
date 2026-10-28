"""
experimento.py — o pipeline inteiro, usado pelo notebook e pelo rodar.py.

  1. dados: HAM10000, split por lesão (treino / val / teste; o teste em duas metades)
  2. para cada arquitetura: treina (ou carrega o checkpoint), escolhe o limiar na validação
  3. para cada arquitetura: lê a rede por dentro no teste (features em profundidade relativa)
  4. deixa-uma-arquitetura-fora: tipo 1 × tipo 2
  5. salva planilha + figuras

Tudo que é caro fica salvo em cfg.saida (checkpoints, limiares, features). Se o notebook cair no meio,
rodar de novo continua de onde parou.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fuzzy_t2 as ft
import leitura_geral as lg
import redes as rd


@dataclass
class Config:
    metadata_csv: str = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"
    img_dirs: list = field(default_factory=lambda: ["/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1",
                                                     "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2"])
    saida: str = "saida_tipo2"
    arquiteturas: list = field(default_factory=lambda: ["efficientnet_b0", "resnet50", "vit_b_16"])
    epocas: int = 15
    bs: int = 32
    paciencia: int = 4
    lado_cache: int = 256
    frac_val: float = 0.10
    frac_teste: float = 0.25
    n_ref: int = 2000                 # imagens do treino usadas para treinar as probes e o banco de vizinhos
    k_features: int = 3               # features por regra (Wang-Mendel)
    k_aditivo: int = 5                # features do fuzzy aditivo, da regressão logística e do boosting
    forcar: tuple = ("prox_p",)       # features que sempre entram (P é a pista mais forte)
    q_alarme: float = 0.20            # fração das imagens de origem acima do limiar do alarme
    n_boot: int = 1000
    pretreinado: bool = True
    pesos_locais: dict = field(default_factory=dict)   # {"resnet50": "/kaggle/input/.../resnet50.pth"} se não houver internet
    lr_backbone: dict = field(default_factory=dict)    # sobrescreve redes.LR_BACKBONE
    max_imagens: int = 0              # >0: usa só uma amostra (teste rápido)
    seed: int = 42


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def preparar_dados(cfg, log=print):
    meta = rd.carregar_ham(cfg.metadata_csv, cfg.img_dirs)
    if cfg.max_imagens and len(meta) > cfg.max_imagens:
        lesoes = meta.lesion_id.drop_duplicates().sample(frac=1, random_state=cfg.seed)
        meta = meta[meta.lesion_id.isin(set(lesoes.iloc[:cfg.max_imagens]))].reset_index(drop=True)
    meta = rd.split_por_lesao(meta, cfg.frac_val, cfg.frac_teste, cfg.seed)
    out = Path(cfg.saida); out.mkdir(parents=True, exist_ok=True)
    arq = out / "split.csv"
    if arq.exists():                                   # garante o MESMO split numa segunda execução
        antigo = pd.read_csv(arq)
        meta = meta.drop(columns=["split", "metade"]).merge(antigo[["image_id", "split", "metade"]], on="image_id")
        log("split: reaproveitado de split.csv")
    else:
        meta[["image_id", "lesion_id", "dx", "is_malignant", "split", "metade"]].to_csv(arq, index=False)
    resumo = meta.groupby("split").agg(imagens=("image_id", "size"), lesoes=("lesion_id", "nunique"),
                                       malignas=("is_malignant", "mean"))
    log(resumo.round(3).to_string())
    cache = rd.carregar_cache(meta.image_path, cfg.lado_cache)
    return meta, cache


def treinar_ou_carregar(cfg, nome, meta, cache, log=print):
    out = Path(cfg.saida); ck, lj = out / f"rede_{nome}.pt", out / f"limiar_{nome}.json"
    dev = _dev()
    rede = rd.criar_rede(nome, cfg.pretreinado and not ck.exists(), cfg.pesos_locais.get(nome))
    idx = {s: np.where(meta.split == s)[0] for s in ("treino", "val", "teste")}
    y = meta.is_malignant.values
    if ck.exists() and lj.exists():
        rede.load_state_dict(torch.load(ck, map_location="cpu")); rede.to(dev).eval()
        info = json.load(open(lj)); log(f"{nome}: checkpoint carregado (limiar {info['limiar']:.3f})")
        hist = pd.read_csv(out / f"historico_{nome}.csv") if (out / f"historico_{nome}.csv").exists() else pd.DataFrame()
        return rede, info, hist
    lr = {**rd.LR_BACKBONE, **cfg.lr_backbone}.get(nome, 1e-4)
    log(f"{nome}: treinando ({len(idx['treino'])} imagens, {cfg.epocas} épocas, lr backbone {lr:g}, {dev})")
    torch.manual_seed(cfg.seed)
    rede, hist = rd.treinar(rede, cache, idx["treino"], y[idx["treino"]], idx["val"], y[idx["val"]], dev,
                            cfg.epocas, cfg.bs, lr, cfg.paciencia, cfg.seed, log)
    p_val = rd.prever(rede, cache, idx["val"], dev)
    lim = rd.escolher_limiar(p_val, y[idx["val"]])
    p_te = rd.prever(rede, cache, idx["teste"], dev)
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    info = {"rede": nome, "limiar": lim, "AUROC val": float(roc_auc_score(y[idx["val"]], p_val)),
            "AUROC teste": float(roc_auc_score(y[idx["teste"]], p_te)),
            "acurácia balanceada teste": float(balanced_accuracy_score(y[idx["teste"]], p_te >= lim)),
            "taxa de erro teste": float(((p_te >= lim) != y[idx["teste"]]).mean()),
            "pontos de leitura": len(rede.pontos), "parâmetros (M)": sum(p.numel() for p in rede.parameters()) / 1e6}
    torch.save(rede.state_dict(), ck); json.dump(info, open(lj, "w"), indent=1); hist.to_csv(out / f"historico_{nome}.csv", index=False)
    log(f"{nome}: AUROC teste {info['AUROC teste']:.3f} | erro {info['taxa de erro teste']:.1%} | limiar {lim:.3f}")
    return rede, info, hist


def ler_ou_carregar(cfg, nome, rede, info, meta, cache, log=print):
    arq = Path(cfg.saida) / f"features_{nome}.csv"
    if arq.exists():
        log(f"{nome}: features carregadas de {arq.name}")
        return pd.read_csv(arq)
    dev = _dev(); t0 = time.time()
    tr = np.where(meta.split == "treino")[0]
    ref = np.random.default_rng(cfg.seed).choice(tr, size=min(cfg.n_ref, len(tr)), replace=False)
    leitor = lg.LeitorGeral(rede, info["limiar"]).fit(cache, ref, meta.is_malignant.values[ref], dev)
    tab = lg.ler(leitor, meta, np.where(meta.split == "teste")[0], cache, dev)
    tab.to_csv(arq, index=False)
    log(f"{nome}: {leitor.B} pontos de leitura | {tab.erro.sum()} erros em {len(tab)} imagens de teste | {time.time() - t0:.0f}s")
    return tab


def figuras(tabs, fous, preds, pasta, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pasta = Path(pasta); arqs = []
    # 1) trajetória média em profundidade relativa, por rede: acertos × erros
    fig, axes = plt.subplots(1, len(tabs), figsize=(4.2 * len(tabs), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    cols = [f"traj_r{j}" for j in range(lg.N_REL)]; x = np.linspace(0, 1, lg.N_REL)
    for ax, (nome, t) in zip(axes, tabs.items()):
        for rot, m, cor in [("acertos", t.erro == 0, "#2a78d6"), ("erros", t.erro == 1, "#c0392b")]:
            if m.sum() == 0:
                continue
            q = np.percentile(t.loc[m, cols].values, [25, 50, 75], axis=0)
            ax.plot(x, q[1], color=cor, lw=2, label=f"{rot} (n={m.sum()})"); ax.fill_between(x, q[0], q[2], color=cor, alpha=.15)
        ax.axhline(.5, color="gray", ls="--", lw=1); ax.set_title(nome, fontsize=10); ax.set_xlabel("profundidade relativa (0 = início, 1 = fim)")
        ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("concorda com a resposta final")
    fig.tight_layout(); a = pasta / "fig_trajetorias.png"; fig.savefig(a, dpi=150); plt.close(fig); arqs.append(a)
    # 2) a faixa do tipo 2 de cada feature (primeiro alvo, features do fuzzy aditivo)
    chave = [k for k in fous if k[1] == "Fuzzy aditivo"][0]; fo = fous[chave]
    feats = list(dict.fromkeys(fo.feature)); fontes = [c.replace("pico em ", "") for c in fo.columns if c.startswith("pico em ")]
    fig, axes = plt.subplots(1, len(feats), figsize=(3.6 * len(feats), 3.2))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, feats):
        sub = fo[fo.feature == f]
        Cs = [sub[f"pico em {s}"].values for s in fontes]
        lo_x, hi_x = min(c.min() for c in Cs), max(c.max() for c in Cs); pad = 0.15 * (hi_x - lo_x + 1e-9)
        xs = np.linspace(lo_x - pad, hi_x + pad, 400)
        Ms = np.stack([ft.pert_t1(xs[:, None], c[None, :])[:, 0, :] for c in Cs])   # (S, n, 3)
        for k, cor in zip(range(3), ["#2a78d6", "#7f8c8d", "#c0392b"]):
            ax.fill_between(xs, Ms[:, :, k].min(0), Ms[:, :, k].max(0), color=cor, alpha=.25, lw=0)
            for s in range(len(Cs)):
                ax.plot(xs, Ms[s, :, k], color=cor, lw=1, ls=["-", "--", ":", "-."][s % 4])
            ax.text(Cs[0][k], 1.04, ft.TERMOS[k], ha="center", fontsize=8, color=cor)
        ax.set_title(f, fontsize=10); ax.set_ylim(0, 1.12); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("pertinência")
    fig.suptitle(f"Rede de fora: {chave[0]} · faixa do tipo 2 = discordância entre {' e '.join(fontes)} (linhas)", fontsize=10)
    fig.tight_layout(); a = pasta / "fig_faixas_tipo2.png"; fig.savefig(a, dpi=150); plt.close(fig); arqs.append(a)
    # 3) largura do intervalo × erro
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (nome, p) in zip(axes, preds.items()):
        ax.scatter(p["Fuzzy aditivo: tipo 2 transferido"], p["Fuzzy aditivo t2 largura"], s=6, alpha=.35,
                   c=np.where(p.erro == 1, "#c0392b", "#2a78d6"))
        ax.set_title(f"rede de fora: {nome} (vermelho = erro)", fontsize=10); ax.set_xlabel("nota de erro (fuzzy aditivo tipo 2)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("largura do intervalo"); fig.tight_layout()
    a = pasta / "fig_intervalos.png"; fig.savefig(a, dpi=150); plt.close(fig); arqs.append(a)
    log("figuras: " + ", ".join(x.name for x in arqs))
    return arqs


def salvar(cfg, infos, hists, tabs, res, zonas, regras, fous, preds):
    """Abas principais = só P, referências não fuzzy e fuzzy aditivo (o que vai para o artigo).
    Wang-Mendel fica em abas 'interno_*', só para consulta."""
    arq = Path(cfg.saida) / "resultados_tipo2.xlsx"
    princ, intern = res[res.método.isin(ft.PRINCIPAIS)], res[res.método.isin(ft.INTERNOS)]
    resumo = princ.pivot_table(index="método", columns="rede alvo", values="AUROC").reindex(ft.PRINCIPAIS)
    with pd.ExcelWriter(arq) as xw:
        pd.Series({k: str(v) for k, v in asdict(cfg).items()}).to_frame("valor").to_excel(xw, sheet_name="config")
        pd.DataFrame(infos).to_excel(xw, sheet_name="redes", index=False)
        resumo.round(4).to_excel(xw, sheet_name="resumo_AUROC")
        princ.round(4).to_excel(xw, sheet_name="deixa_uma_fora", index=False)
        zonas[zonas.sistema.str.startswith("Fuzzy aditivo")].round(4).to_excel(xw, sheet_name="zonas", index=False)
        for (alvo, fam), r in regras.items():
            if fam == "Fuzzy aditivo":
                r.to_excel(xw, sheet_name=f"regras_t2_{alvo}"[:31], index=False)
                fous[(alvo, fam)].round(4).to_excel(xw, sheet_name=f"faixas_{alvo}"[:31], index=False)
        for n, h in hists.items():
            if len(h):
                h.to_excel(xw, sheet_name=f"treino_{n}"[:31], index=False)
        for n, p in preds.items():
            p.drop(columns=[c for c in p.columns if c.startswith("Wang-Mendel")]).to_excel(xw, sheet_name=f"notas_{n}"[:31], index=False)
        # só para consulta
        intern.round(4).to_excel(xw, sheet_name="interno_wm_metricas", index=False)
        zonas[zonas.sistema.str.startswith("Wang-Mendel")].round(4).to_excel(xw, sheet_name="interno_wm_zonas", index=False)
        for (alvo, fam), r in regras.items():
            if fam == "Wang-Mendel":
                r.to_excel(xw, sheet_name=f"interno_wm_regras_{alvo}"[:31], index=False)
    return arq


def rodar_tudo(cfg, log=print):
    t0 = time.time()
    meta, cache = preparar_dados(cfg, log)
    infos, hists, tabs = [], {}, {}
    for nome in cfg.arquiteturas:
        rede, info, hist = treinar_ou_carregar(cfg, nome, meta, cache, log)
        tabs[nome] = ler_ou_carregar(cfg, nome, rede, info, meta, cache, log)
        infos.append(info); hists[nome] = hist
        del rede
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    res, zonas, regras, fous, preds = ft.deixa_uma_fora(tabs, lg.FEATURES, cfg.k_features, cfg.k_aditivo, cfg.forcar,
                                                        cfg.q_alarme, cfg.n_boot, log)
    figs = figuras(tabs, fous, preds, cfg.saida, log)
    arq = salvar(cfg, infos, hists, tabs, res, zonas, regras, fous, preds)
    log(f"pronto em {(time.time() - t0) / 60:.1f} min → {arq}")
    return {"meta": meta, "infos": infos, "hists": hists, "tabs": tabs, "res": res, "zonas": zonas,
            "regras": regras, "fous": fous, "preds": preds, "figuras": figs, "planilha": arq}
