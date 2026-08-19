import os
import sys
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import DB_URL
from models import Base, CorpusSentence, BlunderLabel, Annotation

from data.ingest import run_ingestion
from data.label  import run_labeling

from parse.ccg_parser import build_parser, build_ansatz, parse_sentences, diagrams_to_circuits

from quantum.ansatz   import LABEL_MAP, N_CLASSES
from quantum.train    import train, predict
from quantum.transfer import train_transfer, apply_transfer
from quantum.evaluate import classification_report, plot_training, plot_embeddings


Path("results").mkdir(exist_ok=True)


def load_labeled_sentences(session, limit: int | None = None):
    rows = (
        session.query(CorpusSentence.sentence, BlunderLabel.category)
        .join(Annotation, CorpusSentence.annotation_id == Annotation.id)
        .join(BlunderLabel, BlunderLabel.annotation_id == Annotation.id)
        .filter(BlunderLabel.confidence >= 0.6)
        .all()
    )
    if limit:
        rows = rows[:limit]

    label_index = {v: k for k, v in LABEL_MAP.items()}
    sentences, labels = [], []
    for sentence, category in rows:
        if category in label_index:
            sentences.append(sentence)
            labels.append(label_index[category])
    return sentences, labels


def load_style_pairs(session, limit: int = 40):
    engine_rows = (
        session.query(CorpusSentence.sentence)
        .join(Annotation, CorpusSentence.annotation_id == Annotation.id)
        .join(BlunderLabel, BlunderLabel.annotation_id == Annotation.id)
        .filter(BlunderLabel.category == "tactical_oversight")
        .limit(limit)
        .all()
    )
    gm_rows = (
        session.query(CorpusSentence.sentence)
        .join(Annotation, CorpusSentence.annotation_id == Annotation.id)
        .join(BlunderLabel, BlunderLabel.annotation_id == Annotation.id)
        .filter(BlunderLabel.category == "positional_degradation")
        .limit(limit)
        .all()
    )
    n = min(len(engine_rows), len(gm_rows))
    return [(engine_rows[i].sentence, gm_rows[i].sentence) for i in range(n)]


def split_train_test(sentences, labels, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sentences))
    split = int(len(sentences) * (1 - test_ratio))
    train_idx = idx[:split]
    test_idx  = idx[split:]
    train_s = [sentences[i] for i in train_idx]
    train_l = [labels[i]    for i in train_idx]
    test_s  = [sentences[i] for i in test_idx]
    test_l  = [labels[i]    for i in test_idx]
    return train_s, train_l, test_s, test_l


def main():
    engine_db = create_engine(DB_URL)
    Base.metadata.create_all(engine_db)
    Session = sessionmaker(bind=engine_db)
    session = Session()

    print("\n[1] Ingesting PGN files...")
    run_ingestion()

    print("\n[2] Assigning blunder labels...")
    run_labeling()

    print("\n[3] Loading labeled sentences...")
    sentences, labels = load_labeled_sentences(session, limit=200)
    print(f"    loaded {len(sentences)} labeled sentences")

    if len(sentences) < 10:
        print("    not enough data — add PGN files to pgn_files/ and rerun.")
        sys.exit(1)

    print("\n[4] Parsing sentences (CCG)...")
    parser = build_parser()
    ansatz = build_ansatz()

    train_s, train_l, test_s, test_l = split_train_test(sentences, labels)

    train_diags, train_s = parse_sentences(train_s, parser)
    test_diags,  test_s  = parse_sentences(test_s,  parser)

    train_l = train_l[:len(train_diags)]
    test_l  = test_l[:len(test_diags)]

    train_circuits = diagrams_to_circuits(train_diags, ansatz)
    test_circuits  = diagrams_to_circuits(test_diags,  ansatz)

    print(f"    train: {len(train_circuits)}  test: {len(test_circuits)}")

    print("\n[5] Training blunder classifier...")
    model, clf_history = train(
        train_circuits, train_l,
        test_circuits,  test_l,
        n_epochs=120,
        batch_size=8,
        lr=0.1,
    )

    print("\n[6] Evaluation...")
    classification_report(test_circuits, test_l, model, split="test")
    plot_embeddings(train_circuits, train_l, model)

    print("\n[7] Training style transfer channel...")
    style_pairs = load_style_pairs(session, limit=40)
    session.close()

    if len(style_pairs) < 4:
        print("    not enough style pairs — skipping transfer training.")
        transfer_history = {"infidelity": []}
        phi = None
    else:
        engine_sents = [e for e, _ in style_pairs]
        gm_sents     = [g for _, g in style_pairs]

        engine_diags, engine_sents = parse_sentences(engine_sents, parser)
        gm_diags,     gm_sents    = parse_sentences(gm_sents,     parser)

        n = min(len(engine_diags), len(gm_diags))
        engine_circuits = diagrams_to_circuits(engine_diags[:n], ansatz)
        gm_circuits     = diagrams_to_circuits(gm_diags[:n],    ansatz)

        phi, transfer_history = train_transfer(
            engine_circuits, gm_circuits, model,
            n_layers=3, n_epochs=200, lr=0.05,
        )

    print("\n[8] Plotting results...")
    plot_training(clf_history, transfer_history)

    print("\n[9] Sample inference...")
    samples = [
        "White misses the fork on d5",
        "The pawn structure collapses completely",
        "Black blunders under time pressure",
        "White deviates from mainline theory",
    ]

    sample_diags, parsed = parse_sentences(samples, parser)
    sample_circuits      = diagrams_to_circuits(sample_diags, ansatz)
    results              = predict(sample_circuits, model, top_k=2)

    for sentence, result in zip(parsed, results):
        top = max(result, key=result.get)
        print(f"  \"{sentence}\"")
        print(f"    → {top}  ({result[top]:.3f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
