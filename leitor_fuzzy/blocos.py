"""
blocos.py — acha sozinho onde ler uma rede qualquer (PyTorch), sem código por arquitetura.

Ideia: começa pelos filhos de primeiro nível da rede e vai "abrindo" o maior deles (mais parâmetros)
enquanto couber no limite de pontos de leitura. Depois remove o que não tem parâmetros e a cabeça
(a última camada, que já é a resposta). A ordem final é a ordem em que a rede executa os módulos.
"""
import torch
import torch.nn as nn


def _n_par(m):
    return sum(p.numel() for p in m.parameters())


NORMAS = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm)
CONTAINERS = (nn.Sequential, nn.ModuleList)


def _reais(m):
    """Filhos com parâmetros que não são só normalização."""
    return [c for c in m.children() if _n_par(c) > 0 and not isinstance(c, NORMAS)]


def _composto(c):
    """Um bloco de verdade (residual, MBConv, encoder...), e não uma camada nem um par conv+norm."""
    return any(True for _ in c.children()) and not (isinstance(c, CONTAINERS) and len(_reais(c)) <= 1)


def _pilha(c):
    """Contêiner com 2+ blocos compostos do mesmo tipo: uma pilha de blocos repetidos."""
    if not isinstance(c, CONTAINERS):
        return False
    tipos = [type(k) for k in _reais(c) if _composto(k)]
    return any(tipos.count(t) >= 2 for t in set(tipos))


def _abrivel(m):
    """Abre só PILHAS: um contêiner com 2+ filhos reais, ou um módulo que embrulha uma pilha de blocos
    repetidos (com 80%+ dos parâmetros). O interior de um bloco (conv, bn, conv...) não é aberto."""
    r = _reais(m)
    if isinstance(m, CONTAINERS):
        return len(r) >= 2
    return any(_pilha(c) and _n_par(c) >= 0.8 * _n_par(m) for c in r)


def _ordem_execucao(rede, exemplo):
    """Executa a rede uma vez e devolve {módulo: (posição em que rodou, formato da saída)}."""
    info, hs = {}, []
    for m in rede.modules():
        def gancho(mod, i, o, m=m):
            if m not in info and torch.is_tensor(o):
                info[m] = (len(info), tuple(o.shape))
        hs.append(m.register_forward_hook(gancho))
    try:
        with torch.no_grad():
            saida = rede(exemplo)
    finally:
        for h in hs:
            h.remove()
    return info, tuple(saida.shape)


def achar_blocos(rede: nn.Module, exemplo: torch.Tensor, max_blocos: int = 20):
    """Devolve (blocos, cabeça) na ordem de execução.
    `exemplo`: um lote pequeno de entradas (ex.: 2 imagens), só para descobrir a ordem e os formatos."""
    rede.eval()
    info, fmt_saida = _ordem_execucao(rede, exemplo)
    lista = _reais(rede) or [rede]
    while True:                                   # abre a maior pilha que ainda couber no limite
        aberto = False
        for m in sorted([m for m in lista if _abrivel(m)], key=_n_par, reverse=True):
            novos = _reais(m)
            if len(lista) - 1 + len(novos) <= max_blocos + 1:     # +1: a cabeça sai depois
                i = lista.index(m); lista = lista[:i] + novos + lista[i + 1:]; aberto = True
                break
        if not aberto:
            break
    lista = sorted([m for m in lista if m in info], key=lambda m: info[m][0])
    # cabeça = último módulo executado cuja saída tem o formato da saída da rede
    cabeca = lista[-1] if info[lista[-1]][1] == fmt_saida else None
    blocos = lista[:-1] if cabeca is not None else lista
    if len(blocos) < 2:
        raise ValueError("Achei menos de 2 pontos de leitura; passe os blocos à mão: LeitorFuzzy(rede, blocos=[...]).")
    return blocos, cabeca


def resumo_vetor(o: torch.Tensor):
    """Reduz a saída de um bloco a um vetor por exemplo: média espacial (imagens), média dos tokens (sequências)."""
    o = o.float()
    if o.dim() == 4:
        return o.mean((2, 3))
    if o.dim() == 3:
        return o.mean(1)
    return o.flatten(1)
