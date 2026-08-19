import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from lambeq import NumpyModel

from quantum.ansatz import LABEL_MAP, N_CLASSES


COLORS  = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
MARKERS = ["o", "s", "^", "D"]


def classification_report(
    circuits: list,
    labels: list[int],
    model: NumpyModel,
    split: str = "test",
) -> None:
    preds    = model(circuits)
    pred_cls = np.argmax(preds, axis=1)
    acc      = np.mean(pred_cls == np.array(labels))

    print(f"\n{split} accuracy: {acc:.3f}")
    print(f"{'category':<28} {'correct':>8} {'total':>8} {'recall':>8}")
    print("-" * 56)

    for cls in range(N_CLASSES):
        mask    = [i for i, l in enumerate(labels) if l == cls]
        correct = sum(1 for i in mask if pred_cls[i] == cls)
        total   = len(mask)
        recall  = correct / total if total else 0.0
        print(f"  {LABEL_MAP[cls]:<26} {correct:>8} {total:>8} {recall:>8.3f}")


def plot_training(
    clf_history: dict,
    transfer_history: dict,
    save_path: str = "results/training.png",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(clf_history["loss"], color="#e07b54", lw=1.8)
    axes[0].set_title("Classifier loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)

    axes[1].plot(clf_history["train_acc"], label="train", color="#4c8cbf", lw=1.8)
    axes[1].plot(clf_history["test_acc"],  label="test",  color="#e07b54", lw=1.8, ls="--")
    axes[1].set_title("Classifier accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(transfer_history["infidelity"], color="#6b8e6b", lw=1.8)
    axes[2].set_title("Style transfer infidelity")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  saved → {save_path}")
    plt.close()


def plot_embeddings(
    circuits: list,
    labels: list[int],
    model: NumpyModel,
    save_path: str = "results/embeddings.png",
) -> None:
    embeddings = model(circuits)
    pca        = PCA(n_components=2)
    coords     = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in range(N_CLASSES):
        mask = [i for i, l in enumerate(labels) if l == cls]
        if not mask:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=COLORS[cls], marker=MARKERS[cls],
            label=LABEL_MAP[cls], s=70,
            edgecolors="k", linewidths=0.4,
        )

    ax.set_title("Sentence embeddings (PCA)")
    ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  saved → {save_path}")
    plt.close()
