import numpy as np
from lambeq import NumpyModel, SPSAOptimizer

from quantum.ansatz import N_CLASSES, LABEL_MAP


def one_hot(labels: list[int], n_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), n_classes))
    for i, l in enumerate(labels):
        out[i, l] = 1.0
    return out


def cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> float:
    eps = 1e-9
    return -np.mean(np.sum(targets * np.log(predictions + eps), axis=1))


def accuracy(predictions: np.ndarray, labels: list[int]) -> float:
    return np.mean(np.argmax(predictions, axis=1) == np.array(labels))


def build_model(train_circuits: list, test_circuits: list) -> NumpyModel:
    model = NumpyModel.from_diagrams(train_circuits + test_circuits)
    model.initialise_weights()
    return model


def train(
    train_circuits: list,
    train_labels: list[int],
    test_circuits: list,
    test_labels: list[int],
    n_epochs: int = 120,
    batch_size: int = 8,
    lr: float = 0.1,
) -> tuple[NumpyModel, dict]:

    model = build_model(train_circuits, test_circuits)

    optimizer = SPSAOptimizer(
        model=model,
        loss_fn=cross_entropy,
        hyperparams={"a": lr, "c": 0.06, "A": 0.01 * n_epochs},
    )

    train_oh = one_hot(train_labels, N_CLASSES)
    history  = {"loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(n_epochs):
        idx               = np.random.permutation(len(train_circuits))
        circuits_shuffled = [train_circuits[i] for i in idx]
        oh_shuffled       = train_oh[idx]

        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, len(circuits_shuffled), batch_size):
            batch_c = circuits_shuffled[start : start + batch_size]
            batch_l = oh_shuffled[start : start + batch_size]
            epoch_loss += optimizer.step(batch_c, batch_l)
            n_batches  += 1

        train_preds = model(train_circuits)
        test_preds  = model(test_circuits)

        avg_loss  = epoch_loss / max(n_batches, 1)
        train_acc = accuracy(train_preds, train_labels)
        test_acc  = accuracy(test_preds, test_labels)

        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d} | loss {avg_loss:.4f} | "
                  f"train {train_acc:.3f} | test {test_acc:.3f}")

    return model, history


def predict(sentence_circuits: list, model: NumpyModel, top_k: int = 2) -> list[dict]:
    raw = model(sentence_circuits)
    results = []
    for probs in raw:
        probs = np.exp(probs) / np.sum(np.exp(probs))
        ranked = np.argsort(probs)[::-1][:top_k]
        results.append({LABEL_MAP[i]: float(probs[i]) for i in ranked})
    return results
