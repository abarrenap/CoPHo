#!/bin/bash

# Script para generar grafos con un modelo entrenado

echo "🎨 Generador de Grafos CoPHo"
echo "============================="
echo ""

# Verificar argumentos
if [ $# -eq 0 ]; then
    echo "Uso: bash generate_graphs.sh <ruta_al_checkpoint.ckpt> [num_grafos]"
    echo ""
    echo "Ejemplo:"
    echo "  bash generate_graphs.sh outputs/2026-02-07/14-30-45-dimacs_exp1/checkpoints/last.ckpt 10"
    echo ""
    echo "📦 Checkpoints disponibles:"
    find /Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/outputs -name "*.ckpt" 2>/dev/null | head -5
    exit 1
fi

CHECKPOINT=$1
NUM_GRAPHS=${2:-10}  # Por defecto 10 grafos

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint no encontrado: $CHECKPOINT"
    exit 1
fi

echo "✓ Checkpoint: $CHECKPOINT"
echo "✓ Número de grafos a generar: $NUM_GRAPHS"
echo ""

# Activar ambiente y ejecutar
cd /Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src
source $(conda info --base)/etc/profile.d/conda.sh
conda activate copho

echo "🚀 Generando grafos..."
python main.py \
    --config-name=config_dimacs \
    general.test_only="$CHECKPOINT" \
    general.name="generate_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "✅ Generación completada!"
echo "Los grafos generados se guardan en el directorio del experimento."
