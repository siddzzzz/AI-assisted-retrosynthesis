"""Deep Learning Retrosynthesis Policy Training Pipeline.
Optimized for NVIDIA RTX 3050 4GB GPU (CUDA) with Mixed Precision & Gradient Accumulation.
All checkpoints and models are saved locally in ./models/ within this repository.
"""

import os
import sys
import time
import csv
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

# Configure local workspace paths
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class ChemicalReactionDataset(Dataset):
    """Encodes product molecules into Morgan Fingerprints (ECFP4, 2048-bit) and maps to reaction classes/rules."""

    def __init__(self, csv_file: str, rule_to_idx: Dict[str, int]):
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.rule_to_idx = rule_to_idx

        if not os.path.exists(csv_file):
            print(f"[!] Warning: {csv_file} not found. Run scripts/download_uspto_dataset.py first.")
            return

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rxn_smiles = row.get("rxn_smiles", row.get("reactants>reagents>production", ""))
                rule_id = row.get("rule_id", row.get("class", "0"))
                
                # Split reaction into reactants >> product
                if ">>" in rxn_smiles:
                    _, prod_smi = rxn_smiles.split(">>")
                elif ">" in rxn_smiles:
                    parts = rxn_smiles.split(">")
                    prod_smi = parts[-1]
                else:
                    continue

                mol = Chem.MolFromSmiles(prod_smi.strip())
                if mol is not None:
                    # 2048-bit Morgan Fingerprint (Radius 2 = ECFP4)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    arr = np.zeros((2048,), dtype=np.float32)
                    arr[list(fp.GetOnBits())] = 1.0
                    
                    target_idx = self.rule_to_idx.get(rule_id, 0)
                    self.samples.append((arr, target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, label = self.samples[idx]
        return torch.tensor(fp, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class RetrosynthesisPolicyNetwork(nn.Module):
    """Deep residual neural policy network for predicting retrosynthetic bond disconnections."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, num_classes: int = 30, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_policy_model(
    epochs: int = 25,
    batch_size: int = 32,  # Optimized for 4GB VRAM
    learning_rate: float = 1e-3,
    gradient_accumulation_steps: int = 4,
):
    """Train retrosynthesis model on GPU or CPU with mixed precision."""
    # 1. Device Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"[*] Retrosynthesis Training Engine")
    print(f"[*] Device Selected: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] Available GPU VRAM: {vram_gb:.2f} GB (RTX 3050 Profile Active)")
    print(f"============================================================")

    # 2. Reaction Rule Mapping
    from retro_engine.chem.reaction_rules import ReactionRuleLibrary
    rule_lib = ReactionRuleLibrary()
    rules = rule_lib.get_all_rules()
    rule_to_idx = {r.rule_id: i for i, r in enumerate(rules)}
    num_classes = max(len(rule_to_idx), 1)

    # 3. Ensure Dataset Exists
    dataset_path = os.path.join(DATA_DIR, "uspto_50k.csv")
    if not os.path.exists(dataset_path):
        from scripts.download_uspto_dataset import download_uspto_50k
        download_uspto_50k()

    dataset = ChemicalReactionDataset(dataset_path, rule_to_idx)
    if len(dataset) == 0:
        print("[!] No training samples found. Generating fallback synthetic set...")
        from scripts.download_uspto_dataset import generate_offline_training_dataset
        generate_offline_training_dataset(dataset_path)
        dataset = ChemicalReactionDataset(dataset_path, rule_to_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # 4. Model, Loss, Optimizer, Scaler
    model = RetrosynthesisPolicyNetwork(input_dim=2048, hidden_dim=512, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    print(f"[*] Training started on {len(dataset)} chemical reaction samples for {epochs} epochs...")
    model.train()

    best_loss = float("inf")
    checkpoint_path = os.path.join(MODELS_DIR, "retrosynthesis_policy_rtx3050.pt")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        start_t = time.time()

        optimizer.zero_grad()

        for step, (fps, labels) in enumerate(train_loader):
            fps = fps.to(device)
            labels = labels.to(device)

            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(fps)
                    loss = criterion(outputs, labels) / gradient_accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(fps)
                loss = criterion(outputs, labels) / gradient_accumulation_steps
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / max(total, 1)
        accuracy = (correct / max(total, 1)) * 100.0
        elapsed = time.time() - start_t

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Top-1 Accuracy: {accuracy:.1f}% | Time: {elapsed:.2f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rule_to_idx": rule_to_idx,
                "loss": best_loss,
            }, checkpoint_path)

    print(f"\n[+] Training Complete! Model checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrosynthesis Policy Training for NVIDIA RTX 3050 / CUDA GPU")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro-batch size (32 recommended for 4GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    train_policy_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
    )
