#!/bin/bash

# Script de inicio rápido para entrenar con DIMACS

echo "🚀 CoPHo - DIMACS Dataset Training"
echo "=================================="
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -d "src" ] || [ ! -d "DIMACS" ]; then
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
python3 -c "
import os
files = [f for f in os.listdir('../DIMACS') if f.endswith('.col')]
print(f'   Total de grafos: {len(files)}')
"

echo ""
echo "🤖 Iniciando entrenamiento..."
echo "   Comando: python main.py --config-name=config_dimacs general.name=dimacs_exp"
echo ""

# Ejecutar el entrenamiento
python main.py --config-name=config_dimacs general.name=dimacs_exp

echo ""
echo "✓ Entrenamiento completado"
