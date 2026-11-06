#!/usr/bin/env bash
# rodar_ruido.sh — a curva de ruído no CIFAR-10 (ResNet-32): 10%, 20% e 40% de rótulos de treino trocados.
# O 0% (cifar10) e o 30% (cifar10_ruido30) já estão em resultados/ e são reaproveitados.
#
#   bash rodar_ruido.sh rapido     # 1 semente por nível (3 redes novas) — ~2 h numa GPU boa
#   bash rodar_ruido.sh completo   # 3 sementes por nível (as que já rodaram são reaproveitadas) — ~5–6 h
#   bash rodar_ruido.sh humano     # ruído REAL (CIFAR-10N e CIFAR-100N, rótulos de pessoas), 1 semente — ~3 h
#   bash rodar_ruido.sh humano-completo   # o mesmo com 3 sementes — ~8–9 h
#
#   nohup bash rodar_ruido.sh rapido > log_ruido.txt 2>&1 &
#   tail -f log_ruido.txt
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'sem GPU'; print('GPU:', torch.cuda.get_device_name(0))"
EXPS="cifar10_ruido10 cifar10_ruido20 cifar10_ruido40"
HUM="cifar10n_aggre cifar10n_random1 cifar10n_worst cifar100n"
case "${1:-rapido}" in
  rapido)   python rodar.py --exp $EXPS --sementes 42 ;;
  completo) python rodar.py --exp $EXPS ;;
  humano)   python rodar.py --exp $HUM --sementes 42 ;;
  humano-completo) python rodar.py --exp $HUM ;;
  *) echo "uso: bash rodar_ruido.sh rapido | completo | humano | humano-completo"; exit 1 ;;
esac
python analise_ruido.py
echo "Mande: resultados/ruido.xlsx  resultados/figuras/ruido.png"
