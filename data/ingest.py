import re
import chess.pgn
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import DB_URL, PGN_DIR
from models import Base, Game, Move, Annotation, CorpusSentence


def get_session():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def extract_eval_from_comment(comment: str) -> float | None:
    match = re.search(r"\[%eval\s+([+-]?\d+\.?\d*)\]", comment)
    if match:
        val = float(match.group(1))
        return val * 100 if abs(val) < 50 else val
    return None


def clean_annotation_text(comment: str) -> str:
    text = re.sub(r"\[%[^\]]+\]", "", comment)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for s in raw:
        s = s.strip()
        tokens = s.split()
        if 3 <= len(tokens) <= 12:
            sentences.append(s)
    return sentences


def ingest_pgn_file(pgn_path: Path, session) -> int:
    count = 0
    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            db_game = Game(
                white_player=game.headers.get("White", ""),
                black_player=game.headers.get("Black", ""),
                result=game.headers.get("Result", ""),
            )
            session.add(db_game)
            session.flush()

            node = game
            move_number = 0
            prev_eval = None

            while node.variations:
                next_node = node.variations[0]
                move_number += 1

                db_move = Move(
                    game_id=db_game.id,
                    move_number=move_number,
                    san=next_node.san() if next_node.move else "",
                )
                session.add(db_move)
                session.flush()

                comment = next_node.comment.strip()
                if comment:
                    current_eval = extract_eval_from_comment(comment)
                    eval_drop = None
                    if prev_eval is not None and current_eval is not None:
                        eval_drop = current_eval - prev_eval

                    cleaned = clean_annotation_text(comment)
                    if cleaned:
                        db_ann = Annotation(
                            move_id=db_move.id,
                            text=cleaned,
                        )
                        session.add(db_ann)
                        session.flush()

                        db_move.eval_drop = eval_drop

                        for sentence in split_into_sentences(cleaned):
                            session.add(CorpusSentence(
                                annotation_id=db_ann.id,
                                sentence=sentence,
                            ))

                    if current_eval is not None:
                        prev_eval = current_eval

                node = next_node

            session.commit()
            count += 1

    return count


def run_ingestion():
    session = get_session()
    pgn_files = list(PGN_DIR.glob("*.pgn"))

    if not pgn_files:
        print(f"No PGN files found in {PGN_DIR}. Add .pgn files and rerun.")
        return

    total = 0
    for pgn_path in pgn_files:
        print(f"Ingesting: {pgn_path.name}")
        n = ingest_pgn_file(pgn_path, session)
        total += n
        print(f"  → {n} games processed")

    print(f"\nTotal games ingested: {total}")
    session.close()


if __name__ == "__main__":
    run_ingestion()