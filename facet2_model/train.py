"""Train the covariance prediction MLP on the prepared dataset splits."""
# .venv/bin/python train.py

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Column definitions ────────────────────────────────────────────────────────
TARGET_PREFIX = "cov_chol_"


def get_feature_target_columns(df: pd.DataFrame):
    target_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    feature_cols = [c for c in df.columns if not c.startswith(TARGET_PREFIX)]
    return feature_cols, target_cols


# ── Model ─────────────────────────────────────────────────────────────────────
class CovarianceSurrogateModel(nn.Module):
    """NN predicts Cholesky factors and outputs a 6x6 covariance matrix."""

    def __init__(self, n_inputs: int, n_outputs: int, y_mean=None, y_std=None):
        super().__init__()
        drop = 0.05
        self.backbone = nn.Sequential(
            nn.Linear(n_inputs, 100),
            nn.ELU(),
            nn.Linear(100, 200),
            nn.ELU(),
            nn.Dropout(p=drop),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Dropout(p=drop),
            nn.Linear(200, 300),
            nn.ELU(),
            nn.Dropout(p=drop),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Dropout(p=drop),
            nn.Linear(300, 200),
            nn.ELU(),
            nn.Dropout(p=drop),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Dropout(p=drop),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, n_outputs),
        )

        if y_mean is None:
            y_mean = torch.zeros(n_outputs, dtype=torch.float32)
        if y_std is None:
            y_std = torch.ones(n_outputs, dtype=torch.float32)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict Cholesky factors and produce 6x6 covariance matrix."""
        chol_norm = self.backbone(x)

        # Denormalize Cholesky factors
        chol_raw = chol_norm * self.y_std + self.y_mean

        # The training targets store the lower-triangular Cholesky factors.
        # Reconstruct L first, then form C = L @ L.T.
        batch_size = chol_raw.shape[0]
        L = torch.zeros(
            (batch_size, 6, 6),
            dtype=chol_raw.dtype,
            device=chol_raw.device,
        )

        tril_idx = torch.tril_indices(row=6, col=6, offset=0, device=chol_raw.device)
        L[:, tril_idx[0], tril_idx[1]] = chol_raw
        cov = L @ L.transpose(1, 2)
        
        return cov

    def chol_norm_to_cov(self, chol_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized Cholesky factors to covariance matrices."""
        chol_raw = chol_norm * self.y_std + self.y_mean
        return chol_vectors_to_covariance(chol_raw)


def build_model(n_inputs: int, n_outputs: int, y_mean=None, y_std=None) -> CovarianceSurrogateModel:
    """Factory retained for compatibility with analysis/inference utilities."""
    return CovarianceSurrogateModel(n_inputs, n_outputs, y_mean=y_mean, y_std=y_std)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_split(path: Path, feature_cols, target_cols, x_mean, x_std, y_mean, y_std):
    df = pd.read_csv(path, low_memory=False)
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def chol_vectors_to_covariance(chol_vectors: torch.Tensor) -> torch.Tensor:
    """Convert batch of stored lower-triangular Cholesky vectors to 6x6 covariance."""
    batch_size = chol_vectors.shape[0]
    L = torch.zeros(
        (batch_size, 6, 6),
        dtype=chol_vectors.dtype,
        device=chol_vectors.device,
    )
    tril_idx = torch.tril_indices(row=6, col=6, offset=0, device=chol_vectors.device)
    L[:, tril_idx[0], tril_idx[1]] = chol_vectors
    return L @ L.transpose(1, 2)


class CovarianceAwareLoss(nn.Module):
    """Loss computed in normalized covariance space."""

    def __init__(
        self,
        model: CovarianceSurrogateModel,
        cov_mean: torch.Tensor,
        cov_std: torch.Tensor,
        cov_loss: str = "mse",
    ):
        super().__init__()
        self.model = model
        self.register_buffer("cov_mean", cov_mean)
        self.register_buffer("cov_std", cov_std)
        if cov_loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif cov_loss == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported cov_loss: {cov_loss}")

    def forward(self, pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
        pred_cov = pred_norm
        target_cov = self.model.chol_norm_to_cov(target_norm)
        pred_cov_norm = (pred_cov - self.cov_mean) / self.cov_std
        target_cov_norm = (target_cov - self.cov_mean) / self.cov_std
        return self.loss_fn(pred_cov_norm, target_cov_norm)


# ── Training ───────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    n_samples = 0
    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(X_batch)
            n_samples += len(X_batch)
    return total_loss / n_samples


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the covariance MLP on prepared dataset splits."
    )
    parser.add_argument("--train-csv", default="dataset-train.csv")
    parser.add_argument("--val-csv", default="dataset-val.csv")
    parser.add_argument("--test-csv", default="dataset-test.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="model-output",
        help="Directory to save model checkpoint and scalers (default: model-output)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience in epochs (default: 20; 0 disables)",
    )
    parser.add_argument(
        "--finetune-batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional fine-tuning batch sizes to run sequentially after base training "
            "(e.g. --finetune-batch-sizes 32 8 2)"
        ),
    )
    parser.add_argument(
        "--finetune-epochs-per-stage",
        type=int,
        default=0,
        help="Fine-tuning epochs to run at each batch-size stage (default: 0 disables)",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for fine-tuning stages (default: 1e-4)",
    )
    parser.add_argument(
        "--finetune-lr-decay",
        type=float,
        default=0.5,
        help="Multiply LR by this factor after each fine-tuning stage (default: 0.5)",
    )
    parser.add_argument(
        "--finetune-plateau-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience during fine-tuning stages (default: 5)",
    )
    parser.add_argument(
        "--finetune-min-lr",
        type=float,
        default=1e-6,
        help="Minimum LR for ReduceLROnPlateau during fine-tuning (default: 1e-6)",
    )
    parser.add_argument(
        "--cov-loss",
        choices=["mse", "l1"],
        default="mse",
        help="Covariance-space objective function (default: mse).",
    )
    return parser


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute scalers from training set only ────────────────────────────────
    print(f"[run] Reading training CSV: {args.train_csv}", flush=True)
    train_df = pd.read_csv(args.train_csv, low_memory=False)
    feature_cols, target_cols = get_feature_target_columns(train_df)
    n_inputs = len(feature_cols)
    n_outputs = len(target_cols)
    print(f"[run] Features: {n_inputs}  Targets: {n_outputs}", flush=True)

    if n_outputs != 21:
        raise SystemExit(
            "Covariance-space loss requires 21 Cholesky targets (cov_chol_0..cov_chol_20)."
        )
    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    y_train_raw = train_df[target_cols].values.astype(np.float32)

    x_mean = X_train_raw.mean(axis=0)
    x_std = X_train_raw.std(axis=0)
    x_std[x_std == 0] = 1.0  # avoid divide-by-zero for constant columns

    y_mean = y_train_raw.mean(axis=0)
    y_std = y_train_raw.std(axis=0)
    y_std[y_std == 0] = 1.0

    # Compute covariance-element normalizers from train targets in raw units.
    y_train_raw_t = torch.from_numpy(y_train_raw)
    train_cov = chol_vectors_to_covariance(y_train_raw_t).numpy().reshape(-1, 36)
    cov_mean = train_cov.mean(axis=0).astype(np.float32)
    cov_std = train_cov.std(axis=0).astype(np.float32)
    cov_std[cov_std == 0] = 1.0

    # Save input and output transformers separately
    input_transformers = {
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
        "feature_cols": feature_cols,
    }
    output_transformers = {
        "y_mean": torch.from_numpy(y_mean),
        "y_std": torch.from_numpy(y_std),
        "target_cols": target_cols,
    }
    covariance_transformers = {
        "cov_mean": torch.from_numpy(cov_mean),
        "cov_std": torch.from_numpy(cov_std),
        "cov_labels": [f"cov_{i}{j}" for i in range(6) for j in range(6)],
    }
    torch.save(input_transformers, output_dir / "input_transformers.pt")
    torch.save(output_transformers, output_dir / "output_transformers.pt")
    torch.save(covariance_transformers, output_dir / "covariance_transformers.pt")
    print(f"[run] Input transformers saved to {output_dir}/input_transformers.pt", flush=True)
    print(f"[run] Output transformers saved to {output_dir}/output_transformers.pt", flush=True)
    print(f"[run] Covariance transformers saved to {output_dir}/covariance_transformers.pt", flush=True)

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_ds = load_split(
        Path(args.train_csv), feature_cols, target_cols,
        x_mean, x_std, y_mean, y_std,
    )
    val_ds = load_split(
        Path(args.val_csv), feature_cols, target_cols,
        x_mean, x_std, y_mean, y_std,
    )
    test_ds = load_split(
        Path(args.test_csv), feature_cols, target_cols,
        x_mean, x_std, y_mean, y_std,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ── Model, loss, optimizer ─────────────────────────────────────────────────
    y_mean_t = torch.from_numpy(y_mean).to(device)
    y_std_t = torch.from_numpy(y_std).to(device)
    cov_mean_t = torch.from_numpy(cov_mean).to(device).view(1, 6, 6)
    cov_std_t = torch.from_numpy(cov_std).to(device).view(1, 6, 6)
    model = build_model(n_inputs, n_outputs, y_mean=y_mean_t, y_std=y_std_t).to(device)
    print(f"[run] Model architecture:\n{model}", flush=True)

    criterion = CovarianceAwareLoss(
        model=model,
        cov_mean=cov_mean_t,
        cov_std=cov_std_t,
        cov_loss=args.cov_loss,
    )
    print(f"[run] Loss mode: cov (per-element normalized), objective={args.cov_loss}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n[run] Training for up to {args.epochs} epochs ...", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:04d}/{args.epochs}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
            flush=True,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "model.pt")
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(
                f"[run] Early stopping triggered at epoch {epoch} "
                f"(no improvement for {args.patience} epochs)",
                flush=True,
            )
            break

    # ── Optional staged fine-tuning with smaller batches / lower LR ───────────
    do_finetune = (
        args.finetune_batch_sizes is not None
        and args.finetune_epochs_per_stage > 0
        and len(args.finetune_batch_sizes) > 0
    )
    if do_finetune:
        print(
            "\n[run] Starting fine-tuning stages "
            f"(batch_sizes={args.finetune_batch_sizes}, "
            f"epochs_per_stage={args.finetune_epochs_per_stage}, "
            f"initial_lr={args.finetune_lr:.2e})",
            flush=True,
        )

        # Resume from best base checkpoint before fine-tuning.
        model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
        stage_lr = args.finetune_lr

        for stage_idx, stage_bs in enumerate(args.finetune_batch_sizes, start=1):
            stage_train_loader = DataLoader(train_ds, batch_size=stage_bs, shuffle=True)
            stage_val_loader = DataLoader(val_ds, batch_size=stage_bs)

            stage_optimizer = torch.optim.Adam(model.parameters(), lr=stage_lr)
            stage_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                stage_optimizer,
                mode="min",
                factor=0.5,
                patience=args.finetune_plateau_patience,
                min_lr=args.finetune_min_lr,
            )

            print(
                f"[finetune stage {stage_idx}] batch_size={stage_bs} "
                f"lr={stage_lr:.2e} epochs={args.finetune_epochs_per_stage}",
                flush=True,
            )

            for stage_epoch in range(1, args.finetune_epochs_per_stage + 1):
                t0 = time.time()
                train_loss = run_epoch(
                    model,
                    stage_train_loader,
                    criterion,
                    stage_optimizer,
                    device,
                    train=True,
                )
                val_loss = run_epoch(
                    model,
                    stage_val_loader,
                    criterion,
                    stage_optimizer,
                    device,
                    train=False,
                )
                stage_scheduler.step(val_loss)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                elapsed = time.time() - t0
                print(
                    f"[finetune {stage_idx}:{stage_epoch:03d}/"
                    f"{args.finetune_epochs_per_stage}] "
                    f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                    f"lr={stage_optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
                    flush=True,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), output_dir / "model.pt")

            stage_lr = max(stage_lr * args.finetune_lr_decay, args.finetune_min_lr)

    # ── Final evaluation on test set ───────────────────────────────────────────
    print("\n[run] Loading best checkpoint for test evaluation ...", flush=True)
    model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
    test_loss = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"[run] Test objective loss (cov, {args.cov_loss}): {test_loss:.6f}", flush=True)

    # MAE in original covariance units
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred_cov = model(X_batch.to(device)).cpu().numpy()
            all_preds.append(pred_cov)
            all_targets.append(y_batch.numpy())

    preds_cov = np.concatenate(all_preds)
    targets_chol_raw = np.concatenate(all_targets) * y_std + y_mean
    targets_cov = chol_vectors_to_covariance(torch.from_numpy(targets_chol_raw)).numpy()

    preds_cov_flat = preds_cov.reshape(preds_cov.shape[0], -1)
    targets_cov_flat = targets_cov.reshape(targets_cov.shape[0], -1)
    mae_per_element = np.abs(preds_cov_flat - targets_cov_flat).mean(axis=0)
    mae_overall = mae_per_element.mean()

    print(
        f"[run] Test MAE (covariance units, mean over 36 matrix elements): {mae_overall:.6e}",
        flush=True,
    )
    print(f"[run] Test MAE per covariance element:", flush=True)
    for i in range(6):
        for j in range(6):
            idx = i * 6 + j
            print(f"       cov_{i}{j}: {mae_per_element[idx]:.6e}", flush=True)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    print(f"\n[run] Training history saved to {output_dir}/training_history.csv", flush=True)
    print(f"[run] Model saved to {output_dir}/model.pt", flush=True)


if __name__ == "__main__":
    main()