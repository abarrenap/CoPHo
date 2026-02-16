#!/usr/bin/env python
"""Train a guidance model to predict GIN embeddings for MIS graphs."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src import utils
from src.datasets.mis_dataset import MISDataModule
from src.models.GNN_model import GraphStructModel
from src.models import condi_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train guidance model for embedding condition")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--normalize-emb", action="store_true")
    parser.add_argument("--save-dir", type=str, default="/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src/weights")
    parser.add_argument("--save-name", type=str, default="CLASSIFIER_struct_embedding_community.pth")
    return parser.parse_args()


def get_batch_y(batch, normalize_emb: bool):
    y = batch.y
    if y.dim() == 3 and y.size(1) == 1:
        y = y.squeeze(1)
    elif y.dim() == 1:
        y = y.unsqueeze(0)
    if normalize_emb:
        y = F.normalize(y, p=2, dim=-1)
    return y


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data using MISDataModule
    class SimpleCfg:
        class dataset:
            datadir = "/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/data/mis"
        class train:
            batch_size = 32
            num_workers = 0
        class general:
            gpus = 0
            name = "guidance_embedding_train"

    cfg = SimpleCfg()
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

    datamodule = MISDataModule(cfg)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = GraphStructModel(
        in_dim=1,
        hidden_dim=condi_config.HIDDEN_DIM,
        num_layers=condi_config.NUM_LAYERS,
        dropout=condi_config.DROPOUT,
        out_dim=70,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            dense, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            adj = dense.E[..., 1]
            y = get_batch_y(batch, args.normalize_emb)

            pred = model(adj, node_mask)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        avg_loss = running / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for batch in val_loader:
                batch = batch.to(device)
                dense, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                adj = dense.E[..., 1]
                y = get_batch_y(batch, args.normalize_emb)
                pred = model(adj, node_mask)
                val_running += F.mse_loss(pred, y).item()
            val_loss = val_running / max(len(val_loader), 1)

        print(f"Epoch {epoch}/{args.epochs} - train {avg_loss:.6f} - val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / args.save_name
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} (val {best_val:.6f})")

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch} (best {best_epoch}, val {best_val:.6f})")
            break


if __name__ == "__main__":
    main()
