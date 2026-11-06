#!/usr/bin/env bash
# rodar_novo.sh — roda as pistas NOVAS (fronteira entre classes) em cima das redes JÁ TREINADAS em resultados/.
# Não retreina nada: acrescenta as pistas com uma passada rápida na referência e refaz só a comparação.
#
#   bash rodar_novo.sh rapido     # 3 cenários × 1 semente (CIFAR com ruído, CIFAR-100, HAM todas) — ~30–60 min
#   bash rodar_novo.sh completo   # todos os cenários e sementes — algumas horas
#
# Para não perder o progresso se o terminal fechar:
#   nohup bash rodar_novo.sh rapido > log_novo.txt 2>&1 &
#   tail -f log_novo.txt
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
MODO=${1:-rapido}
ls resultados/*/semente_*/modelo.pt >/dev/null 2>&1 || { echo "ERRO: não achei redes treinadas em resultados/. Rode antes: bash rodar_tudo.sh"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(), 'sem GPU'; print('GPU:', torch.cuda.get_device_name(0))"

case "$MODO" in
  rapido)
    echo "===== RÁPIDO: 3 cenários × semente 42 — $(date) ====="
    python rodar.py --exp cifar10_ruido30 cifar100_vgg16 ham10000_todas --sementes 42 --refazer avaliacao ;;
  completo)
    echo "===== COMPLETO: todos os cenários e sementes — $(date) ====="
    python rodar.py --refazer avaliacao ;;
  *) echo "uso: bash rodar_novo.sh rapido | completo"; exit 1 ;;
esac

echo "===== relatório e análises — $(date) ====="
python relatorio.py
python analise_confiantes.py --frac 0.25 0.5
python analise_pares.py --pasta resultados --relatorio resultados/relatorio.xlsx

echo "===== PRONTO — $(date) ====="
echo "Mande: resultados/relatorio.xlsx  resultados/confiantes.xlsx  resultados/pares.xlsx"
