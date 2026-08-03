from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from lambeq import BobcatParser, AtomicType, IQPAnsatz
from lambeq.backend.grammar import Diagram

from config import DB_URL, MIN_SENTENCE_TOKENS, MAX_SENTENCE_TOKENS
from models import Base, CorpusSentence


N = AtomicType.NOUN
S = AtomicType.SENTENCE


def build_parser() -> BobcatParser:
    return BobcatParser(verbose="suppress")


def build_ansatz(n_noun_qubits: int = 1, n_sentence_qubits: int = 2) -> IQPAnsatz:
    return IQPAnsatz(
        ob_map={N: n_noun_qubits, S: n_sentence_qubits},
        n_layers=1,
        discard=False,
    )


def is_valid_sentence(text: str) -> bool:
    tokens = text.strip().split()
    return MIN_SENTENCE_TOKENS <= len(tokens) <= MAX_SENTENCE_TOKENS


def parse_sentences(
    sentences: list[str],
    parser: BobcatParser,
) -> tuple[list[Diagram], list[str]]:
    valid_texts = [s for s in sentences if is_valid_sentence(s)]
    diagrams = parser.sentences2diagrams(valid_texts, tokenised=False)

    parsed, texts = [], []
    for diag, text in zip(diagrams, valid_texts):
        if diag is not None:
            parsed.append(diag)
            texts.append(text)

    return parsed, texts


def diagrams_to_circuits(diagrams: list[Diagram], ansatz: IQPAnsatz) -> list:
    return [ansatz(d) for d in diagrams]


def load_corpus_sentences(session) -> list[str]:
    rows = session.query(CorpusSentence.sentence).all()
    return [r.sentence for r in rows]


def run_parsing(
    n_noun_qubits: int = 1,
    n_sentence_qubits: int = 2,
    limit: int | None = None,
) -> tuple[list, list[str]]:
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    sentences = load_corpus_sentences(session)
    session.close()

    if limit:
        sentences = sentences[:limit]

    print(f"Sentences loaded from DB: {len(sentences)}")

    parser = build_parser()
    ansatz = build_ansatz(n_noun_qubits, n_sentence_qubits)

    diagrams, parsed_texts = parse_sentences(sentences, parser)
    print(f"Successfully parsed:      {len(diagrams)} / {len(sentences)}")

    circuits = diagrams_to_circuits(diagrams, ansatz)
    print(f"Circuits compiled:        {len(circuits)}")

    return circuits, parsed_texts


if __name__ == "__main__":
    circuits, texts = run_parsing(limit=50)

    print("\nSample parsed sentences:")
    for text in texts[:5]:
        print(f"  → {text}")