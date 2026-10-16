"""
rodar.py — roda TODO o trabalho: para cada experimento e cada semente, treina a rede, lê a rede por dentro,
compara todos os métodos e faz as análises extras. Salva tudo em resultados/<experimento>/semente_<s>/.

É RETOMÁVEL: se cair no meio, rode de novo o mesmo comando — o que já foi feito (modelo, features,
avaliação) é carregado do disco e não é refeito.

Exemplos:
  python rodar.py                                   # todos os experimentos de config.py, todas as sementes
  python rodar.py --exp ham10000                    # só o HAM10000
  python rodar.py --exp ham10000 cifar10 --sementes 42
  python rodar.py --refazer avaliacao               # refaz só a comparação (sem retreinar nem reler a rede)
  python rodar.py --teste-rapido                    # testa o código em minutos, com dados falsos, sem GPU
Depois: python relatorio.py
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import config as C


def _json(obj, caminho):
    caminho.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def rodar_um(exp, semente, disp, refazer):
    import comparacao as cmp
    import dados
    import extras as ex
    import redes
    from leitura import LeitorRede

    pasta = C.SAIDA / exp / f"semente_{semente}"
    pasta.mkdir(parents=True, exist_ok=True)
    arq_log = open(pasta / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); arq_log.write(msg + "\n"); arq_log.flush()

    e = C.EXPERIMENTOS[exp]; t0 = time.time()
    log(f"\n=== {exp} | semente {semente} | {e['artigo']} | {time.strftime('%d/%m %H:%M')} ===")
    torch.manual_seed(semente); np.random.seed(semente)
    conj = dados.carregar(exp, semente)
    log(f"  teste: {len(conj.teste)} imagens | referência: {len(conj.treino_ref)} | classes: {conj.n_classes}")
    if "divisao" in conj.extras:
        conj.extras["divisao"].to_csv(pasta / "divisao_lesoes.csv", index=False)

    # 1) rede ------------------------------------------------------------------------------------------
    rede = redes.criar(e["arq"], conj.n_classes).to(disp)
    if (pasta / "modelo.pt").exists() and refazer not in ("tudo", "rede"):
        rede.load_state_dict(torch.load(pasta / "modelo.pt", map_location=disp))
        info = json.loads((pasta / "info_rede.json").read_text())
        log("  rede carregada do disco")
    else:
        log("  treinando a rede...")
        info = redes.treinar(rede, conj, e["arq"], semente, disp, log)
        torch.save(rede.state_dict(), pasta / "modelo.pt")
    rede.eval()
    lote = C.LOTE[e["arq"]]
    probs = redes.prever(rede, dados.carregador(conj.teste, lote), disp=disp)
    dec = (probs[:, 1] >= info["limiar"]).astype(int) if conj.binario else probs.argmax(1)
    info.update({"experimento": exp, "semente": semente, "arquitetura": e["arq"], "artigo": e["artigo"],
                 "imagens de teste": len(dec), "acurácia teste": float((dec == conj.y_teste).mean()),
                 "erros da rede": int((dec != conj.y_teste).sum())})
    if conj.binario:
        from sklearn.metrics import roc_auc_score
        info.update({"AUROC da rede (maligna)": float(roc_auc_score(conj.y_teste, probs[:, 1])),
                     "sensibilidade": float(dec[conj.y_teste == 1].mean()), "especificidade": float(1 - dec[conj.y_teste == 0].mean())})
    _json(info, pasta / "info_rede.json")
    log(f"  acurácia {info['acurácia teste']:.3f} | erros {info['erros da rede']}")

    # 2) leitura da rede -------------------------------------------------------------------------------
    if (pasta / "features.csv.gz").exists() and refazer not in ("tudo", "rede", "leitura"):
        tab = pd.read_csv(pasta / "features.csv.gz"); ext = dict(np.load(pasta / "extras.npz"))
        log("  features carregadas do disco")
    else:
        log("  lendo a rede por dentro (referência)...")
        leitor = LeitorRede(rede, conj.n_classes, info["limiar"], conj.vistas, disp, lote).fit(conj.treino_ref, conj.y_ref, log)
        log("  lendo a rede por dentro (teste, todas as vistas)...")
        tab, ext = leitor.transform(conj.teste, log)
        tab["rotulo"] = conj.y_teste
        tab["erro"] = (tab.decisao.values != conj.y_teste).astype(int)
        tab = pd.concat([conj.meta_teste.reset_index(drop=True).drop(columns=["rotulo"], errors="ignore"), tab], axis=1)
        if exp.startswith("cifar10") and e["dados"] == "cifar10" and not C.TESTE_RAPIDO:
            try:
                tab["duvida_humana"] = ex.cifar10h({"x": np.zeros(len(tab))})[1]
            except Exception as err:
                log(f"  (CIFAR-10H indisponível: {err})")
        tab.to_csv(pasta / "features.csv.gz", index=False)
        np.savez_compressed(pasta / "extras.npz", **ext)

    # 3) comparação -------------------------------------------------------------------------------------
    if (pasta / "metricas.csv").exists() and refazer not in ("tudo", "rede", "leitura", "avaliacao"):
        log("  avaliação já feita (use --refazer avaliacao para refazer)")
    else:
        log("  comparando todos os métodos (validação cruzada)...")
        oof, escolhas, fixos = cmp.avaliar(tab, ext, conj.binario, info["limiar"], conj.y_teste, semente)
        y = tab.erro.values
        pd.DataFrame(oof).assign(erro=y).to_csv(pasta / "escores.csv.gz", index=False)
        met = cmp.tabela(y, oof, semente=semente); met.insert(0, "semente", semente); met.insert(0, "experimento", exp)
        rej, rc, roc = cmp.curvas(y, oof)
        rej.to_csv(pasta / "curva_rejeicao.csv", index=False); rc.to_csv(pasta / "curva_risco_cobertura.csv", index=False)
        roc.to_csv(pasta / "curva_roc.csv", index=False)
        import fuzzy as fz
        from leitura import CLASSICAS, NOSSAS
        cand = [c for c in NOSSAS + CLASSICAS if c in tab and tab[c].nunique() > 1]
        fz.diagnostico(tab, cand, escolhas["NOSSO: Wang-Mendel"] + escolhas["NOSSO: Fuzzy aditivo"]).to_csv(pasta / "diagnostico_features.csv", index=False)
        rwm, rad, est = cmp.regras_finais(tab, fixos)
        rwm.to_csv(pasta / "regras_wang_mendel.csv", index=False); rad.to_csv(pasta / "regras_fuzzy_aditivo.csv", index=False)
        _json({**est, "escolhas por fold": escolhas}, pasta / "regras_info.json")

        # 4) análises extras ------------------------------------------------------------------------------
        if e["dados"] == "ham10000":
            meta = conj.meta_teste.reset_index(drop=True)
            ex.ham_dx_type(tab, meta, oof).to_csv(pasta / "ham_dx_type.csv", index=False)
            fe = conj.extras["fotos_extras"]
            if len(fe):
                pe = redes.prever(rede, dados.carregador(fe, lote), disp=disp)
                ex.ham_multifotos(tab, meta, oof, conj.extras["fotos_extras_meta"], (pe[:, 1] >= info["limiar"]).astype(int)) \
                  .to_csv(pasta / "ham_multifotos.csv", index=False)
        if e["dados"] == "cifar10" and not C.TESTE_RAPIDO:
            try:
                ex.cifar10h(oof)[0].to_csv(pasta / "cifar10h.csv", index=False)
            except Exception as err:
                log(f"  (CIFAR-10H indisponível: {err})")
        met.to_csv(pasta / "metricas.csv", index=False)           # gravado por último: marca a avaliação como completa
        m = met.set_index("método")
        for k in ["Só P", "Softmax (MSP)", "MC Dropout (desvio)", "Chen et al. 2019"] + cmp.NOSSOS:
            log(f"    {k:28s} AUROC {m.loc[k, 'AUROC']:.3f} | E-AURC {m.loc[k, 'E-AURC (x1000)']:.1f} | "
                f"acurácia encaminhando 20% {m.loc[k, 'acurácia encaminhando 20%']:.3f}")
    log(f"  pronto em {(time.time() - t0) / 60:.1f} min")
    arq_log.close()


def preparar_teste_rapido():
    """Dados falsos e configurações mínimas para testar o código inteiro em poucos minutos na CPU."""
    from PIL import Image
    C.TESTE_RAPIDO = True
    C.SAIDA = C.SAIDA.parent / "resultados_teste_rapido"
    C.HAM_DIR = C.SAIDA.parent / "dados_falsos" / "ham10000"
    C.N_REF, C.N_BOOT, C.T_MC, C.HAM_N_LESOES, C.TRABALHADORES = 200, 50, 5, 300, 0
    if not (C.HAM_DIR / "HAM10000_metadata.csv").exists():
        r = np.random.default_rng(0); linhas = []; k = 0
        for p in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
            (C.HAM_DIR / p).mkdir(parents=True, exist_ok=True)
        for les in range(450):
            d = r.choice(["nv", "mel", "bkl", "bcc", "akiec", "df", "vasc"], p=[.6, .12, .1, .06, .04, .04, .04])
            for _ in range(r.integers(1, 3)):
                cor = np.array([200, 110, 100]) if d in ("mel", "bcc", "akiec") else np.array([180, 140, 120])
                Image.fromarray(np.clip(r.normal(cor, 40, (64, 64, 3)), 0, 255).astype(np.uint8)) \
                     .save(C.HAM_DIR / f"HAM10000_images_part_{1 + k % 2}" / f"ISIC_{k:07d}.jpg")
                linhas.append({"lesion_id": f"HAM_{les:07d}", "image_id": f"ISIC_{k:07d}", "dx": d,
                               "dx_type": r.choice(["histo", "consensus", "follow_up"])}); k += 1
        pd.DataFrame(linhas).to_csv(C.HAM_DIR / "HAM10000_metadata.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", default=None, help="experimentos (padrão: config.EXPERIMENTOS_PADRAO)")
    ap.add_argument("--sementes", nargs="+", type=int, default=None)
    ap.add_argument("--refazer", choices=["tudo", "rede", "leitura", "avaliacao"], default=None)
    ap.add_argument("--teste-rapido", action="store_true")
    a = ap.parse_args()
    if a.teste_rapido:
        preparar_teste_rapido()
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {disp}" + ("" if disp == "cuda" else "  (sem GPU: só faz sentido com --teste-rapido)"))
    if disp == "cuda":
        torch.backends.cudnn.benchmark = True
    for exp in a.exp or C.EXPERIMENTOS_PADRAO:
        for s in a.sementes or C.EXPERIMENTOS[exp].get("sementes", C.SEMENTES):
            rodar_um(exp, s, disp, a.refazer)
    print("\nTudo pronto. Agora rode:  python relatorio.py")
