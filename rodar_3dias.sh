#!/usr/bin/env bash
# rodar_3dias.sh — rodada longa: vários datasets × as mesmas 4 redes (ResNet-32, VGG-16, MobileNetV2, CNN pequena).
# Datasets (config.GRADE_DADOS, em ordem de prioridade):
#   CIFAR-10N pior de 3 (40% de rótulos errados), CIFAR-10 (0%), CIFAR-100N (40%), CIFAR-10N 1 pessoa (17%),
#   CIFAR-10N voto de 3 (9%), CIFAR-100 (0%), Fashion-MNIST, SVHN, MNIST e ANIMAL-10N (só se dados/animal10n existir).
# Primeiro TODOS com a semente 42, depois todos com a 7, depois com a 2026.
# Pode parar a qualquer hora (Ctrl+C ou desligar): o que terminou fica salvo, e rodar de novo continua de onde parou.
# Depois de cada semente, a análise geral é refeita (resultados/geral.xlsx e resultados/insights.txt).
#
#   nohup bash rodar_3dias.sh > log_3dias.txt 2>&1 &
#   tail -f log_3dias.txt
#   bash rodar_3dias.sh --sementes 42      # só uma semente
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'sem GPU'; print('GPU:', torch.cuda.get_device_name(0))" \
  || { echo "ERRO: o PyTorch não vê a GPU (ative o .venv?)"; exit 1; }
echo "disco livre: $(df -h . | awk 'NR==2{print $4}')"

SEMENTES="42 7 2026"
[ "$1" = "--sementes" ] && { shift; SEMENTES="$*"; }

EXPS=$(python -c "
import config as C
from pathlib import Path
tem_animal = sum(1 for _ in Path(C.ANIMAL_DIR).rglob('*.jpg')) > 50000 if Path(C.ANIMAL_DIR).exists() else False
print(' '.join(e for e in C.EXPERIMENTOS_GRADE if tem_animal or C.GRADE_INFO[e][0] != 'animal10n'))")
echo "experimentos ($(echo $EXPS | wc -w)): $EXPS"

for S in $SEMENTES; do
  echo "===== semente $S — início $(date) ====="
  python rodar.py --exp $EXPS --sementes $S
  echo "===== semente $S — fim $(date) ====="
  python analise_geral.py || echo "(análise geral falhou; os resultados continuam salvos)"
done

python relatorio.py --todos || true
echo "===== PRONTO — $(date) ====="
echo "Mande: resultados/geral.xlsx  resultados/insights.txt  resultados/figuras/geral_*.png"
