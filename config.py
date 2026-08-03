from pathlib import Path

BASE_DIR = Path(__file__).parent

DB_PATH = BASE_DIR / "chess_qnlp.db"
DB_URL  = f"sqlite:///{DB_PATH}"

PGN_DIR = BASE_DIR / "pgn_files"
PGN_DIR.mkdir(exist_ok=True)

BLUNDER_THRESHOLDS = {
    "tactical_oversight":    -200,
    "positional_degradation": -80,
    "time_pressure":          -50,
    "opening_mistake":        -30,
}

MIN_SENTENCE_TOKENS = 3
MAX_SENTENCE_TOKENS = 12