# 🎉 Ambiente CoPHo Configurado

Tu ambiente `copho` está completamente configurado y listo para usar. Aquí está el resumen:

## ✅ Instalación Completada

- ✓ Python 3.9
- ✓ PyTorch 2.0.1 (Mac - CPU)
- ✓ PyTorch Geometric 2.3.1
- ✓ PyTorch Lightning 2.0.4
- ✓ graph-tool 2.45
- ✓ Hydra 1.3.2
- ✓ Todos los requirements instalados
- ✓ CoPHo instalado en modo editable
- ✓ orca compilado

## 📊 Dataset DIMACS

Tu dataset DIMACS está listo:
- **Total de grafos:** 58
- **Train:** 37 grafos
- **Val:** 9 grafos
- **Test:** 12 grafos

## 🚀 Cómo Entrenar el Modelo

### Opción 1: Entrenar desde cero con DIMACS
```bash
cd /Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src
conda activate copho
python main.py --config-name=config_dimacs general.name=dimacs_exp1
```

### Opción 2: Entrenar con parámetros personalizados
```bash
python main.py --config-name=config_dimacs \
  general.name=dimacs_v2 \
  train.batch_size=16 \
  train.learning_rate=1e-4 \
  train.epochs=500
```

### Opción 3: Entrenar con más verbosidad
```bash
python main.py --config-name=config_dimacs \
  general.name=dimacs_debug \
  hydra.run.dir=../outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}-${general.name}
```

## 🧪 Cómo Probar el Modelo

Una vez entrenado, prueba con:
```bash
cd /Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src
conda activate copho
python main.py --config-name=config_dimacs general.test_only=path/to/checkpoint.ckpt
```

## 📁 Ubicaciones Importantes

- **Código:** `/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src/`
- **Datos DIMACS:** `/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/DIMACS/`
- **Configuraciones:** `/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/configs/`
- **Dataset loader:** `/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src/datasets/dimacs_dataset.py`
- **Outputs:** `/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/outputs/`

## ⚙️ Parámetros Disponibles

### Generales
- `general.name` - Nombre del experimento
- `general.epochs` - Número de épocas (default: 1000)

### Entrenamiento
- `train.batch_size` - Tamaño del batch (default: 32)
- `train.learning_rate` - Learning rate (default: 1e-3)
- `train.epochs` - Épocas (default: 1000)
- `train.num_workers` - Workers para data loading (default: 4)

### Modelo
- `model.num_layers` - Capas del modelo (default: 9)
- `model.hidden_dims` - Dimensiones ocultas (default: [256])

## 🔍 Ver Resultados

Los resultados se guardan en:
```
outputs/YYYY-MM-DD/HH-MM-SS-{experiment_name}/
├── checkpoints/
├── logs/
└── generated_graphs/
```

## 📝 Notas

- El modelo se ejecuta en **CPU** por defecto (Mac)
- Si tienes GPU disponible, edita las configuraciones de modelo
- Los checkpoints se guardan automáticamente cada epoch
- Los logs se registran con wandb si está configurado

## 💡 Próximos Pasos

1. Activa el ambiente: `conda activate copho`
2. Navega a src: `cd /Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src`
3. Comienza el entrenamiento: `python main.py --config-name=config_dimacs general.name=mi_experimento`

¡Listo para entrenar! 🚀
