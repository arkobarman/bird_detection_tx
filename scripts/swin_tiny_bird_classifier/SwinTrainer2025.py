"""
Swin-Tiny trainer for seabird classification.

Takes a label_mapping.json and a crops directory — no hardcoded experiment presets.
The split CSVs (train.csv, val.csv, test.csv) are expected to live in the same
directory as the label_mapping.json.

Label mapping format
--------------------
A flat JSON object mapping raw species codes to training class labels:

    {
        "ROTEA": "ROTEA_ROTEF",
        "ROTEF": "ROTEA_ROTEF",
        "SATEA": "SATEA_SATEF",
        "BRPEC": "BRPEC",
        "AMAVA": "OTHERS",
        ...
    }

- Keys are the raw species_name values in the annotation CSVs.
- Values are the remapped_label used for training.
- Multiple keys sharing the same value are merged into one class.
- The set of unique values becomes the class list (sorted alphabetically).

Split CSV format
----------------
Each CSV (train.csv, val.csv, test.csv) must contain at least:
    crop_path       - path to the crop image, relative to --image-root/crops/
    remapped_label  - training class label (must match values in label_mapping.json)

Usage
-----
    python -m scripts.swin_tiny_bird_classifier.SwinTrainer2025 \\
        --label-mapping splits/my_split/label_mapping.json \\
        --image-root data/BirdDataset_2025_05052026_crops/crops

    python -m scripts.swin_tiny_bird_classifier.SwinTrainer2025 \\
        --label-mapping splits/my_split/label_mapping.json \\
        --image-root data/BirdDataset_2025_05052026_crops/crops \\
        --epochs 50 --lr 5e-5 --out-dir runs/my_run
"""

import time
from datetime import timedelta
from pathlib import Path
import json
import argparse
from typing import Optional
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
import timm
from torch import nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    classification_report,
    f1_score,
    average_precision_score,
)
from .DataLoader import set_up_data_loaders


# ============================================================================
# DEFAULT HYPERPARAMETERS
# ============================================================================

MODEL_NAME = "swin_tiny_patch4_window7_224"
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 0.01
ACCUM_STEPS = 1
WARMUP_EPOCHS = 10
BATCH_TRAIN = 32
BATCH_EVAL = 128
NUM_WORKERS = 16
INPUT_SIZE = 224
USE_SAMPLER = True
SAMPLER_POWER = 0.5  # 1/sqrt(count): softer correction for long-tailed classes
AMP = True

HARDWARE = "1x NVIDIA A10 (Lambda)"

# Set in train()
LOG_PATH = None


# ============================================================================
# SPLIT CONFIG
# ============================================================================

def load_split_config(label_mapping_path: Path) -> dict:
    """Load label_mapping.json and resolve sibling train/val/test CSVs.

    Returns a dict with keys:
        classes    - sorted list of unique training class labels
        train_csv  - Path to train.csv
        val_csv    - Path to val.csv
        test_csv   - Path to test.csv
        split_name - name of the parent directory (used for run naming)
    """
    with open(label_mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    classes = sorted(set(mapping.values()))
    split_dir = label_mapping_path.parent

    for csv_name in ("train.csv", "val.csv", "test.csv"):
        p = split_dir / csv_name
        if not p.exists():
            raise FileNotFoundError(f"Expected split CSV not found: {p}")

    return {
        "classes": classes,
        "train_csv": split_dir / "train.csv",
        "val_csv": split_dir / "val.csv",
        "test_csv": split_dir / "test.csv",
        "split_name": split_dir.name,
    }


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str) -> None:
    print(message)
    if LOG_PATH is not None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


# ============================================================================
# PLOTTING
# ============================================================================

def plot_curves(history, path):
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)

    ax_loss.plot(ep, history["train_loss"], label="train_loss", color="blue")
    ax_loss.plot(ep, history["val_loss"], label="val_loss", color="red")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_acc.plot(ep, history["train_acc"], label="train_acc", color="blue")
    ax_acc.plot(ep, history["val_acc"], label="val_acc", color="red")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_two_cms(y1, p1, y2, p2, classes, path, titles=("Validation", "Test")):
    from sklearn.metrics import confusion_matrix
    labels = list(range(len(classes)))

    cm1 = confusion_matrix(y1, p1, labels=labels)
    cm2 = confusion_matrix(y2, p2, labels=labels)

    def _row_normalize(cm):
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.divide(cm, row_sums, where=row_sums > 0) * 100.0
        pct[np.isnan(pct)] = 0.0
        return pct

    cm1_pct = _row_normalize(cm1)
    cm2_pct = _row_normalize(cm2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    vmin, vmax = 0.0, 100.0
    im = None
    for ax, cm_pct, title in zip(axes, (cm1_pct, cm2_pct), titles):
        im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        thresh = (cm_pct.max() + cm_pct.min()) / 2.0 if cm_pct.size else 0.0
        for i in range(cm_pct.shape[0]):
            for j in range(cm_pct.shape[1]):
                val = cm_pct[i, j]
                text = f"{val:.1f}" if val > 0 else "0.0"
                color = "white" if val > thresh else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=6)

    cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.046, pad=0.02)
    cbar.set_label("Percentage (%)")
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ============================================================================
# EVALUATION
# ============================================================================

def eval_collect(model, dl, num_classes, device, amp_enabled):
    model.eval()
    y_true, y_pred, probs = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(xb)
                p = torch.softmax(logits, dim=1)
            probs.append(p.cpu().numpy())
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    probs = np.concatenate(probs, axis=0) if probs else np.zeros((0, num_classes), dtype=np.float32)
    return y_true, y_pred, probs


def compute_map_ovr(y_true, probs, num_classes):
    if len(y_true) == 0:
        return 0.0, np.zeros(num_classes)
    y_true = np.array(y_true)
    Y = np.zeros((len(y_true), num_classes), dtype=np.int32)
    Y[np.arange(len(y_true)), y_true] = 1
    ap_per_class = []
    for k in range(num_classes):
        try:
            ap = average_precision_score(Y[:, k], probs[:, k])
        except ValueError:
            ap = 0.0
        ap_per_class.append(ap if np.isfinite(ap) else 0.0)
    return float(np.mean(ap_per_class)), np.array(ap_per_class)


def evaluate_full(model, dl, classes, header, save_prefix, out_dir, device, amp_enabled):
    y_true, y_pred, probs = eval_collect(model, dl, len(classes), device, amp_enabled)

    labels = list(range(len(classes)))
    log(f"\n{header}:")
    log(classification_report(y_true, y_pred, labels=labels, target_names=classes, digits=3, zero_division=0))
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    log(f"{header} macro-F1: {macro_f1:.3f}")

    mAP_macro, ap_cls = compute_map_ovr(y_true, probs, len(classes))
    log(f"{header} mAP (macro, one-vs-rest): {mAP_macro:.3f}")

    per_class_ap = {cls: float(ap) for cls, ap in zip(classes, ap_cls.tolist())}
    with (out_dir / f"{save_prefix}_ap_per_class.json").open("w", encoding="utf-8") as f:
        json.dump(per_class_ap, f, indent=2)
    log(f"{header} per-class AP written to {save_prefix}_ap_per_class.json")

    ds = dl.dataset
    misclassified = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != yp:
            row = ds.rows[i]
            misclassified.append({
                "crop_path": row["crop_path"],
                "true_label": classes[yt],
                "predicted_label": classes[yp],
                "true_class_conf": round(float(probs[i][yt]), 4),
                "predicted_class_conf": round(float(probs[i][yp]), 4),
            })
    with open(out_dir / f"{save_prefix}_misclassified.json", "w", encoding="utf-8") as f:
        json.dump(misclassified, f, indent=2)
    log(f"{header}: {len(misclassified)} misclassified saved to {save_prefix}_misclassified.json")

    return {"macro_f1": float(macro_f1), "map_macro": float(mAP_macro), "n_samples": int(len(y_true))}, y_true, y_pred


def split_composition(ds, classes, label_key):
    cnt = Counter([r[label_key] for r in ds.rows])
    return {c: int(cnt.get(c, 0)) for c in classes}


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(
    label_mapping: Path,
    image_root: Path,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    accum_steps: int = ACCUM_STEPS,
    warmup_epochs: int = WARMUP_EPOCHS,
    batch_train: int = BATCH_TRAIN,
    batch_eval: int = BATCH_EVAL,
    num_workers: int = NUM_WORKERS,
    model_name: str = MODEL_NAME,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    amp_enabled: bool = AMP,
    out_dir: Optional[Path] = None,
):
    """Train Swin-Tiny from a label_mapping.json + crops directory.

    Args:
        label_mapping: Path to label_mapping.json. train.csv, val.csv, test.csv
                       must exist in the same directory.
        image_root:    Root directory containing the cropped images referenced
                       by crop_path in the split CSVs.
        out_dir:       Output directory for this run. Defaults to
                       runs/<split_name>/.
    """
    global LOG_PATH

    run_start = time.perf_counter()

    cfg = load_split_config(label_mapping)
    split_name = cfg["split_name"]
    classes = cfg["classes"]
    num_classes = len(classes)

    if out_dir is None:
        out_dir = Path("runs") / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    lr_str = f"{int(round(lr * 1e6)):04d}"
    wd_str = f"{int(round(weight_decay * 10000)):04d}"
    run_name = f"swin_ep{epochs}_lr{lr_str}_wd{wd_str}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    LOG_PATH = run_dir / "train.log"
    ckpt_path = run_dir / "best.pt"

    log(f"[info] Split:         {split_name}")
    log(f"[info] Label mapping: {label_mapping}")
    log(f"[info] Image root:    {image_root}")
    log(f"[info] Run directory: {run_dir}")
    log(f"[info] Classes ({num_classes}): {classes}")

    dl_train, dl_val, dl_test, meta = set_up_data_loaders(
        train_csv=cfg["train_csv"],
        val_csv=cfg["val_csv"],
        test_csv=cfg["test_csv"],
        image_root=image_root,
        label_key="remapped_label",
        input_size=INPUT_SIZE,
        use_sampler=USE_SAMPLER,
        batch_train=batch_train,
        batch_eval=batch_eval,
        num_workers=num_workers,
        max_per_class=None,
        merge_groups=None,
        sampler_power=SAMPLER_POWER,
    )

    dl_cfg = meta["dataloader_config"]

    log("=" * 60)
    log("TRAINING CONFIGURATION")
    log("=" * 60)
    log(f"  split:            {split_name}")
    log(f"  model:            {model_name}")
    log(f"  loss:             CrossEntropy")
    log(f"  epochs:           {epochs}")
    log(f"  warmup_epochs:    {warmup_epochs}")
    log(f"  lr:               {lr}")
    log(f"  weight_decay:     {weight_decay}")
    log(f"  accum_steps:      {accum_steps}")
    log(f"  AMP:              {amp_enabled}")
    log(f"  device:           {device}")
    log(f"  hardware:         {HARDWARE}")
    log(f"  input_size:       {dl_cfg['input_size']}")
    log(f"  batch_train:      {dl_cfg['batch_train']}")
    log(f"  batch_eval:       {dl_cfg['batch_eval']}")
    log(f"  use_sampler:      {dl_cfg['use_sampler']}")
    log(f"  sampler_power:    {dl_cfg['sampler_power']}  (1.0=1/count, 0.5=1/sqrt(count), 0.0=uniform)")
    log(f"  num_workers:      {dl_cfg['num_workers']}")
    log(f"  image_root:       {dl_cfg['image_root']}")
    log(f"  num_classes:      {num_classes}")
    log(f"  classes:          {classes}")
    log(f"  train_size:       {meta['sizes']['train']}")
    log(f"  val_size:         {meta['sizes']['val']}")
    log(f"  test_size:        {meta['sizes']['test']}")
    log("=" * 60)

    comp_train = split_composition(dl_train.dataset, classes, "remapped_label")
    comp_val   = split_composition(dl_val.dataset,   classes, "remapped_label")
    comp_test  = split_composition(dl_test.dataset,  classes, "remapped_label")
    with open(run_dir / "split_composition.json", "w", encoding="utf-8") as f:
        json.dump({"train": comp_train, "val": comp_val, "test": comp_test}, f, indent=2)
    log("[info] saved split_composition.json")

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(epochs - warmup_epochs, 1))
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_metric = float("-inf")

    for ep in range(1, epochs + 1):
        ep_start = time.perf_counter()

        if ep <= warmup_epochs:
            lr_now = lr * ep / max(warmup_epochs, 1)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        else:
            sched.step()
            lr_now = opt.param_groups[0]["lr"]

        model.train()
        running_loss = running_correct = running_count = 0
        opt.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(dl_train, start=1):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb)

            scaler.scale(loss / accum_steps).backward()
            if step % accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                pred = logits.argmax(1)
                running_correct += (pred == yb).sum().item()
                running_count += yb.size(0)
                running_loss += loss.item() * yb.size(0)

        train_acc = running_correct / max(1, running_count)
        train_loss = running_loss / max(1, running_count)

        t_val = time.perf_counter()
        model.eval()
        v_loss = v_correct = v_count = 0
        y_val_true, y_val_pred = [], []

        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(device_type="cuda", enabled=amp_enabled):
                    logits = model(xb)
                    loss = nn.functional.cross_entropy(logits, yb)
                v_loss += loss.item() * yb.size(0)
                preds = logits.argmax(1)
                v_correct += (preds == yb).sum().item()
                v_count += yb.size(0)
                y_val_true.extend(yb.cpu().tolist())
                y_val_pred.extend(preds.cpu().tolist())

        val_acc = v_correct / max(1, v_count)
        val_loss = v_loss / max(1, v_count)
        labels = list(range(num_classes))
        val_macro_f1 = f1_score(y_val_true, y_val_pred, labels=labels, average="macro", zero_division=0)
        t_ep_val = time.perf_counter() - t_val
        ep_dt = time.perf_counter() - ep_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log(
            f"ep {ep:02d} | "
            f"train acc {train_acc:.3f} loss {train_loss:.3f} | "
            f"val acc {val_acc:.3f} loss {val_loss:.3f} | macroF1 {val_macro_f1:.3f} | "
            f"lr {lr_now:.2e} | val_time {t_ep_val:.1f}s | ep_time {ep_dt:.1f}s"
        )

        if val_macro_f1 > best_metric:
            best_metric = val_macro_f1
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "name": model_name,
                "label_mapping": str(label_mapping),
                "split_name": split_name,
            }, ckpt_path)
            log(f"[info] new best val macro-F1 {best_metric:.3f} at epoch {ep:02d}")

    plot_curves(history, run_dir / "curves.pdf")
    log("[info] saved curves.pdf")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    val_summary, val_y, val_p = evaluate_full(
        model, dl_val, classes, "Validation report", "val", run_dir, device, amp_enabled
    )
    test_summary, test_y, test_p = evaluate_full(
        model, dl_test, classes, "Test report", "test", run_dir, device, amp_enabled
    )

    plot_two_cms(val_y, val_p, test_y, test_p, classes, run_dir / "val_test_cms.pdf")
    log("[info] saved val_test_cms.pdf")

    summary = {
        "split_name": split_name,
        "label_mapping": str(label_mapping),
        "model": model_name,
        "epochs": epochs,
        "num_classes": num_classes,
        "classes": classes,
        "best_val_macro_f1": float(best_metric),
        "val_summary": val_summary,
        "test_summary": test_summary,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_time = time.perf_counter() - run_start
    log(f"[info] total run time: {str(timedelta(seconds=int(total_time)))}")

    return summary


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Swin-Tiny seabird classifier from a label_mapping.json split"
    )
    parser.add_argument(
        "--label-mapping", type=str, required=True,
        help="Path to label_mapping.json. train.csv/val.csv/test.csv must be in the same directory."
    )
    parser.add_argument(
        "--image-root", type=str, required=True,
        help="Root directory containing cropped images (crop_path in CSVs is relative to this)."
    )
    parser.add_argument("--epochs",        type=int,   default=EPOCHS)
    parser.add_argument("--lr",            type=float, default=LR)
    parser.add_argument("--weight-decay",  type=float, default=WEIGHT_DECAY)
    parser.add_argument("--accum-steps",   type=int,   default=ACCUM_STEPS)
    parser.add_argument("--warmup-epochs", type=int,   default=WARMUP_EPOCHS)
    parser.add_argument("--batch-train",   type=int,   default=BATCH_TRAIN)
    parser.add_argument("--batch-eval",    type=int,   default=BATCH_EVAL)
    parser.add_argument("--num-workers",   type=int,   default=NUM_WORKERS)
    parser.add_argument("--model-name",    type=str,   default=MODEL_NAME)
    parser.add_argument("--device",        type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir",       type=str,   default=None)
    parser.add_argument("--no-amp",        action="store_true", help="Disable automatic mixed precision")

    args = parser.parse_args()

    train(
        label_mapping=Path(args.label_mapping),
        image_root=Path(args.image_root),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accum_steps=args.accum_steps,
        warmup_epochs=args.warmup_epochs,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        num_workers=args.num_workers,
        model_name=args.model_name,
        device=args.device,
        amp_enabled=not args.no_amp,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
