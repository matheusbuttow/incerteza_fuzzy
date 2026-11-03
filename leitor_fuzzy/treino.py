"""treino.py — treino simples de uma rede de classificação (opcional: a biblioteca também aceita uma rede já treinada)."""
import torch
import torch.nn.functional as Fn

from .leitor import _lotes


def treinar_rede(rede, dados_treino, epocas=20, lr=1e-3, peso_decaimento=1e-4, dispositivo=None, log=print):
    """Adam + entropia cruzada (ou BCE se a rede tem uma saída só). Devolve a rede em modo eval."""
    disp = dispositivo or next(rede.parameters()).device
    opt = torch.optim.Adam(rede.parameters(), lr=lr, weight_decay=peso_decaimento)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    for ep in range(epocas):
        rede.train(); tot, n = 0.0, 0
        for x, y in _lotes(dados_treino, 128):
            x, y = x.to(disp), torch.as_tensor(y).to(disp)
            s = rede(x)
            perda = Fn.binary_cross_entropy_with_logits(s[:, 0], y.float()) if s.shape[1] == 1 else Fn.cross_entropy(s, y.long())
            opt.zero_grad(); perda.backward(); opt.step()
            tot += perda.item() * len(x); n += len(x)
        sched.step()
        if log and (ep + 1) % max(1, epocas // 5) == 0:
            log(f"  época {ep + 1}/{epocas}: perda {tot / n:.4f}")
    return rede.eval()
