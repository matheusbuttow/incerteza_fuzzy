#!/usr/bin/env bash
# rodar_tudo.sh — roda o experimento inteiro, em ordem:
#   1. pipeline principal: 8 cenários, fuzzy aditivo TIPO 1 (+ três zonas), todos os métodos de comparação
#   2. relatório (resultados/relatorio.xlsx)
#   3. fuzzy TIPO 1 × TIPO 2 entre arquiteturas no HAM10000 (EfficientNet-B0, ResNet-50, ViT-B/16)
#
# Uso (dentro da pasta do projeto):
#   bash rodar_tudo.sh                       # máquina nova: treina tudo
#   bash rodar_tudo.sh --refazer avaliacao   # redes já treinadas em resultados/: só refaz a comparação
#
# Para não perder o progresso se o terminal fechar:
#   nohup bash rodar_tudo.sh > log_tudo.txt 2>&1 &
#   tail -f log_tudo.txt
# Se cair no meio, rode de novo o mesmo comando: o que já foi feito é reaproveitado.
# Se faltar memória na GPU ("CUDA out of memory") no tipo 2:  BS_TIPO2=8 bash rodar_tudo.sh
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate

for d in dados/ham10000 dados/archive; do
  if [ -f "$d/HAM10000_metadata.csv" ]; then HAM="$d"; break; fi
done
if [ -z "$HAM" ]; then
  echo "ERRO: não achei dados/ham10000/HAM10000_metadata.csv (nem dados/archive/). Coloque o HAM10000 lá."; exit 1
fi
IMGS=$(ls -d "$HAM"/HAM10000_images_part_* 2>/dev/null | tr '\n' ' ')
if [ -z "$IMGS" ]; then IMGS=$(ls -d "$HAM"/ham10000_images_part_* 2>/dev/null | tr '\n' ' '); fi

# ---- conferências antes de começar ----
NJPG=$(find "$HAM" -name "*.jpg" | sed 's#.*/##' | sort -u | wc -l)
echo "HAM10000: $HAM ($NJPG imagens diferentes; o esperado é 10015)"
[ "$NJPG" -lt 10000 ] && { echo "ERRO: faltam imagens do HAM10000 — confira o unzip."; exit 1; }
MEM=$(python -c "import torch; assert torch.cuda.is_available(), 'sem GPU'; p=torch.cuda.get_device_properties(0); print(p.name, '|', round(p.total_memory/2**30))" ) || { echo "ERRO: o PyTorch não vê a GPU (ative o .venv?)"; exit 1; }
echo "GPU: ${MEM} GB"
GB=${MEM##*| }
BS_TIPO2=${BS_TIPO2:-$([ "$GB" -lt 10 ] && echo 16 || echo 32)}      # ViT com lote 32 precisa de ~10 GB
echo "lote do tipo 2 (EfficientNet/ResNet-50/ViT): $BS_TIPO2"
echo "disco livre: $(df -h . | awk 'NR==2{print $4}') (precisa de uns 15 GB)"

echo "===== 1/3 pipeline principal (tipo 1) — $(date) ====="
python rodar.py "$@"

echo "===== 2/3 relatório — $(date) ====="
python relatorio.py

echo "===== 3/3 tipo 1 × tipo 2 entre arquiteturas (HAM10000) — $(date) ====="
(cd experimentos/ham_tipo2 && python rodar.py --metadata "../../$HAM/HAM10000_metadata.csv" \
    --imagens $(for i in $IMGS; do echo "../../$i"; done) --saida ../../resultados_tipo2 --bs "$BS_TIPO2")

echo "===== PRONTO — $(date) ====="
echo "Mande estes dois arquivos:"
echo "  resultados/relatorio.xlsx"
echo "  resultados_tipo2/resultados_tipo2.xlsx"
