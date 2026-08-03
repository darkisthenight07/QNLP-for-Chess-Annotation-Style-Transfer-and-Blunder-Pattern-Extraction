import re
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import DB_URL, BLUNDER_THRESHOLDS
from models import Base, Annotation, BlunderLabel


TACTICAL_KEYWORDS = {
    "fork", "pin", "skewer", "back rank", "discovered",
    "deflection", "decoy", "overloaded", "mating net", "blunder"
}

POSITIONAL_KEYWORDS = {
    "weak pawn", "isolated", "backward", "outpost", "bishop pair",
    "pawn structure", "passive", "kingside", "queenside", "space"
}

TIME_KEYWORDS = {
    "time trouble", "zeitnot", "clock", "seconds", "quickly",
    "rushed", "increment", "pressure"
}

OPENING_KEYWORDS = {
    "theory", "novelty", "variation", "opening", "mainline",
    "transposition", "deviation", "preparation", "trap"
}


def keyword_category(text: str) -> str | None:
    lower = text.lower()
    scores = {
        "tactical_oversight":    sum(1 for k in TACTICAL_KEYWORDS    if k in lower),
        "positional_degradation": sum(1 for k in POSITIONAL_KEYWORDS if k in lower),
        "time_pressure":         sum(1 for k in TIME_KEYWORDS         if k in lower),
        "opening_mistake":       sum(1 for k in OPENING_KEYWORDS      if k in lower),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def eval_drop_category(eval_drop: float) -> str:
    if eval_drop <= BLUNDER_THRESHOLDS["tactical_oversight"]:
        return "tactical_oversight"
    if eval_drop <= BLUNDER_THRESHOLDS["positional_degradation"]:
        return "positional_degradation"
    if eval_drop <= BLUNDER_THRESHOLDS["time_pressure"]:
        return "time_pressure"
    if eval_drop <= BLUNDER_THRESHOLDS["opening_mistake"]:
        return "opening_mistake"
    return None


def assign_category(annotation_text: str, eval_drop: float | None) -> tuple[str | None, float]:
    kw_cat   = keyword_category(annotation_text)
    eval_cat = eval_drop_category(eval_drop) if eval_drop is not None else None

    if kw_cat and eval_cat:
        if kw_cat == eval_cat:
            return kw_cat, 0.95
        return eval_cat, 0.70

    if eval_cat:
        return eval_cat, 0.80

    if kw_cat:
        return kw_cat, 0.60

    return None, 0.0


def run_labeling():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    annotations = session.query(Annotation).all()
    labeled = 0
    skipped = 0

    for ann in annotations:
        move = ann.move
        eval_drop = getattr(move, "eval_drop", None)

        category, confidence = assign_category(ann.text, eval_drop)

        if category is None:
            skipped += 1
            continue

        existing = session.query(BlunderLabel).filter_by(annotation_id=ann.id).first()
        if existing:
            existing.category   = category
            existing.confidence = confidence
        else:
            session.add(BlunderLabel(
                annotation_id=ann.id,
                category=category,
                confidence=confidence,
            ))
        labeled += 1

    session.commit()
    session.close()

    print(f"Labeled:  {labeled}")
    print(f"Skipped:  {skipped}")
    print(f"Total:    {labeled + skipped}")


def label_distribution(session) -> dict:
    rows = session.query(BlunderLabel.category).all()
    dist = {}
    for (cat,) in rows:
        dist[cat] = dist.get(cat, 0) + 1
    return dist


if __name__ == "__main__":
    run_labeling()

    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    s = Session()
    dist = label_distribution(s)
    s.close()

    print("\nLabel distribution:")
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30} {count}")