import argparse
from typing import Optional

import torch
from tqdm import tqdm

from generate_conditioned_graphs import (
    load_cfg_from_checkpoint,
    build_datamodule_and_infos,
    build_model,
    save_generated_graphs,
)


def generate_graphs(
    checkpoint_path: str,
    output_path: str,
    guidance_path: Optional[str] = None,
    device: str = "cpu",
    num_samples: int = 32,
):
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    device_t = torch.device(device)

    cfg, _ = load_cfg_from_checkpoint(checkpoint_path)
    _, model_kwargs = build_datamodule_and_infos(cfg)

    model = build_model(cfg, model_kwargs, checkpoint_path, guidance_path=guidance_path)
    model.to(device_t)
    model.eval()

    # This checkpoint was trained WITH conditions, so the model architecture
    # expects the embedding dimension. We pass zeros as a "neutral" embedding
    # (no specific graph structure to condition on). This is NOT the same as
    # conditional generation - it's filling the required input dimension.
    # For truly unconditional generation, you'd need a checkpoint trained without conditions.
    cond_dim = int(model_kwargs["dataset_infos"].output_dims.get("y", 0))
    neutral_embedding = None
    if cond_dim > 0:
        neutral_embedding = torch.zeros((1, cond_dim), device=device_t)
        print(f"Model expects embedding dimension {cond_dim}, using neutral (zeros) embedding")

    print(f"Number of samples to generate: {num_samples}")

    samples = []
    samples_left_to_generate = num_samples
    samples_left_to_save = num_samples
    chains_left_to_save = int(cfg.general.chains_to_save)
    batch_id = 0

    with tqdm(total=num_samples, desc="Generating graphs", unit="graph") as pbar:
        while samples_left_to_generate > 0:
            bs = int(cfg.train.batch_size)
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)

            # Prepare batch embedding if needed
            y_cond_batch = None
            cond_batch = None
            if neutral_embedding is not None:
                y_cond_batch = neutral_embedding.expand(to_generate, -1).contiguous()
                cond_batch = y_cond_batch

            molecule_list, _, _ = model.sample_batch(
                batch_id=batch_id,
                batch_size=to_generate,
                num_nodes=None,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=int(cfg.general.number_chain_steps),
                y_cond=y_cond_batch,
                cond=cond_batch,
            )
            samples.extend(molecule_list)

            batch_id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

            pbar.update(to_generate)

    save_generated_graphs(samples, output_path)
    print(f"Saved {len(samples)} graphs to {output_path}")





if __name__ == "__main__":

    checkpoint = "/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/outputs/2026-03-10/11-00-32-mixed_size/checkpoints/mixed_size/last.ckpt"

    output = "../generated/mixed_last_uncond.txt"
    guidance = None #'/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src/weights/CLASSIFIER_struct_embedding_community.pth'
    device = "cpu"
    samples = 100

    generate_graphs(
        checkpoint_path=checkpoint,
        output_path=output,
        guidance_path=guidance,
        device=device,
        num_samples=samples,
    )
