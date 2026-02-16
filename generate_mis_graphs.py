#!/usr/bin/env python
"""
Script para generar grafos similares a los del dataset MIS
usando el modelo entrenado Persist Homo
"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path


def generate_graphs(checkpoint_path: str, num_graphs: int = 100, output_dir: str = None):
    """
    Genera grafos usando un modelo entrenado
    
    Args:
        checkpoint_path: Ruta al checkpoint del modelo entrenado
        num_graphs: Número de grafos a generar
        output_dir: Directorio para guardar los grafos generados
    """
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint no encontrado en {checkpoint_path}")
        return False
    
    # Crear comando de ejecución
    cmd = [
        "python", "src/main.py",
        "--config-name=config_mis_persisthomo",
        f"general.name=mis_ph_generation",
        f"general.test_only={checkpoint_path}",
        f"general.final_model_samples_to_generate={num_graphs}",
        f"general.final_model_samples_to_save={num_graphs}"
    ]
    
    if output_dir:
        cmd.append(f"hydra.run.dir={output_dir}")
    
    print("🚀 Generando grafos similares a MIS...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Grafos a generar: {num_graphs}")
    print(f"   Comando: {' '.join(cmd)}")
    print()
    
    os.system(" ".join(cmd))
    

def list_checkpoints(model_name: str = "mis_persisthomo_exp"):
    """
    Lista los checkpoints disponibles de un modelo entrenado
    """
    ckpt_dir = f"checkpoints/{model_name}"
    
    if not os.path.exists(ckpt_dir):
        print(f"❌ No se encontraron checkpoints en {ckpt_dir}")
        return []
    
    ckpts = sorted(Path(ckpt_dir).glob("*.ckpt"))
    
    if not ckpts:
        print(f"❌ No hay archivos .ckpt en {ckpt_dir}")
        return []
    
    print(f"✓ Checkpoints disponibles en {ckpt_dir}:")
    for i, ckpt in enumerate(ckpts, 1):
        size_mb = os.path.getsize(ckpt) / (1024**2)
        print(f"   {i}. {ckpt.name} ({size_mb:.1f} MB)")
    
    return ckpts


def compare_generated_graphs():
    """
    Compara los grafos generados con los originales
    usando las métricas disponibles
    """
    print("\n📊 Comparando grafos generados con originales...")
    print("   Ejecutar: python tools/evaluation.py")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generar grafos similares a MIS usando modelo Persist Homo"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Ruta al checkpoint del modelo entrenado"
    )
    
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=100,
        help="Número de grafos a generar (default: 100)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="mis_persisthomo_exp",
        help="Nombre del modelo (para listar checkpoints)"
    )
    
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="Listar checkpoints disponibles"
    )
    
    args = parser.parse_args()
    
    # Cambiar a directorio raíz del proyecto
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    if args.list_checkpoints:
        # Listar checkpoints disponibles
        list_checkpoints(args.model_name)
    elif args.checkpoint:
        # Generar grafos con checkpoint específico
        generate_graphs(args.checkpoint, args.num_graphs)
        compare_generated_graphs()
    else:
        # Si no se especifica checkpoint, listar opciones
        print("🎯 Uso:")
        print()
        print("Opción 1: Listar checkpoints disponibles")
        print(f"  python {Path(__file__).name} --list-checkpoints")
        print()
        print("Opción 2: Generar grafos con un checkpoint")
        print(f"  python {Path(__file__).name} --checkpoint checkpoints/mis_persisthomo_exp/epoch=9.ckpt")
        print()
        print("Listar checkpoints disponibles:")
        list_checkpoints(args.model_name)
