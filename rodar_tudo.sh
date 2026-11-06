#!/usr/bin/env bash
# rodar_tudo.sh — roda a seleção de datasets do artigo (config.EXPERIMENTOS_PADRAO), o relatório e as análises.
#   um cenário por domínio: CIFAR-10 limpo, CIFAR-10N "worst" (rótulos humanos), ANIMAL-10N (web), Chaoyang (médico)
#
# Uso (dentro da pasta do projeto):
#   bash rodar_tudo.sh                       # treina o que falta (o que já existe em resultados/ é reaproveitado)
#   bash rodar_tudo.sh --refazer avaliacao   # refaz só a comparação, sem retreinar
#   bash rodar_tudo.sh --sementes 42         # rodada rápida: os 4 cenários com uma semente
#
#   nohup bash rodar_tudo.sh > log_tudo.txt 2>&1 &
#   tail -f log_tudo.txt
# Um dataset que falhar (ex.: arquivos faltando) é avisado e pulado; os outros seguem.
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate

# ---- conferências antes de começar ----
MEM=$(python -c "import torch; assert torch.cuda.is_available(), 'sem GPU'; p=torch.cuda.get_device_properties(0); print(p.name, '|', round(p.total_memory/2**30))") \
  || { echo "ERRO: o PyTorch não vê a GPU (ative o .venv?)"; exit 1; }
echo "GPU: ${MEM} GB"
echo "disco livre: $(df -h . | awk 'NR==2{print $4}')"
N_ANIMAL=$(find dados/animal10n -name "*.jpg" 2>/dev/null | wc -l)
N_CHAO=$(find dados/chaoyang -name "*.json" 2>/dev/null | wc -l)
echo "ANIMAL-10N: $N_ANIMAL imagens em dados/animal10n (esperado ~55.000)"
echo "Chaoyang:   $N_CHAO arquivos .json em dados/chaoyang (esperado train.json e test.json)"
[ "$N_ANIMAL" -lt 50000 ] && echo "  AVISO: ANIMAL-10N incompleto — esse dataset será pulado."
[ "$N_CHAO" -lt 2 ] && echo "  AVISO: Chaoyang não encontrado — esse dataset será pulado."
echo "(CIFAR, MNIST e os rótulos do CIFAR-N são baixados sozinhos.)"

echo "===== 1/2 todos os datasets — $(date) ====="
python rodar.py "$@"

echo "===== 2/2 relatório e análises — $(date) ====="
python relatorio.py
python analise_confiantes.py --frac 0.25 0.5
python analise_ruido.py || true

echo "===== PRONTO — $(date) ====="
echo "Mande: resultados/relatorio.xlsx  resultados/confiantes.xlsx  resultados/ruido.xlsx"
