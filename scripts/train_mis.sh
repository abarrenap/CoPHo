#!/bin/bash

# Script de inicio rápido para entrenar con MIS

echo "🚀 CoPHo - MIS Dataset Training"
echo "=================================="
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -d "src" ] || [ ! -d "data/mis" ]; then
    echo "❌ Error: Ejecuta este script desde la carpeta raíz de CoPHo"
    echo "   Ubicación esperada: /Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/"
    exit 1
fi

# Activar ambiente conda
echo "📦 Activando ambiente conda 'copho'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate copho

if [ $? -ne 0 ]; then
    echo "❌ Error: No se puede activar el ambiente 'copho'"
    echo "   Crea el ambiente con: bash setup_environment.sh"
    exit 1
fi

# Navegar a src
cd src

echo "✓ Ambiente activado"
echo ""
echo "📊 Información del Dataset:"
echo "   Dataset: MIS (Maximum Independent Set)"
echo "   Ubicación: ../data/mis/"

echo ""
echo "🤖 Iniciando entrenamiento..."
echo "   Comando: python main.py --config-name=config_mis general.name=mis_exp"
echo ""

# Ejecutar el entrenamiento
python main.py --config-name=config_mis general.name=mis_exp

echo ""
echo "✓ Entrenamiento completado"
