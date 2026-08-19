# QNLP for Chess Annotation Style Transfer and Blunder Pattern Extraction

A quantum natural language processing pipeline that classifies chess blunder types from grandmaster commentary and transfers annotation style from engine output to human GM language — using the DisCoCat compositional framework, IQP quantum circuits, and nearest-neighbour quantum embedding retrieval.

---

## Folder Structure

```
chess_qnlp/
│
├── main.py                    ← run this to execute the full pipeline
├── config.py                  ← all paths, thresholds, and constants
├── models.py                  ← SQLAlchemy ORM (5 database tables)
├── requirements.txt
├── README.md
│
├── pgn_files/                 ← PUT YOUR .pgn FILES HERE
│   └── (your files).pgn
│
├── results/                   ← plots are saved here automatically
│   ├── training.png
│   └── embeddings.png
│
├── chess_qnlp.db              ← SQLite database (auto-created on first run)
│
├── data/
│   ├── __init__.py
│   ├── ingest.py              ← reads PGN files, writes Game/Move/Annotation rows to DB
│   ├── label.py               ← assigns blunder category + confidence to each annotation
│   └── preprocess.py          ← converts chess notation to plain English before parsing
│
├── parse/
│   ├── __init__.py
│   └── ccg_parser.py          ← CCG parsing → string diagrams → IQP circuits
│
└── quantum/
    ├── __init__.py
    ├── ansatz.py              ← qubit counts, label map, ansatz builder
    ├── device.py              ← GPU-aware device selector (lightning → CPU fallback)
    ├── train.py               ← NumpyModel + SPSA classifier training loop
    ├── transfer.py            ← style transfer unitary U(φ) via PennyLane
    ├── decode.py              ← nearest-neighbour retrieval decoder (quantum state → text)
    └── evaluate.py            ← accuracy report, PCA embeddings plot, training curves
```

---

## What Each File Does

### Root

| File | Purpose |
|------|---------|
| `main.py` | Runs all 11 pipeline steps in order. Entry point for everything. |
| `config.py` | Single source of truth for DB path, PGN directory, blunder centipawn thresholds, sentence length limits. Edit this to tune the pipeline. |
| `models.py` | SQLAlchemy ORM defining 5 tables: `games`, `moves`, `annotations`, `corpus_sentences`, `blunder_labels`. Do not edit unless adding new columns. |

### `data/`

| File | Purpose |
|------|---------|
| `ingest.py` | Opens every `.pgn` in `pgn_files/`, walks the game tree node by node, extracts Stockfish `[%eval ...]` tags, computes centipawn eval drop between consecutive moves, splits annotation text into sentences, and writes everything to SQLite. |
| `label.py` | Reads annotations from the DB, scores them against keyword lists (tactical, positional, time, opening) and against eval-drop thresholds, combines both signals into a category + confidence score, writes a `BlunderLabel` row per annotation. |
| `preprocess.py` | Converts chess-specific tokens to English before CCG parsing. `Nf3!` becomes `knight moves to f3`, `Bxf7+` becomes `bishop captures on f7`, `??` becomes `blunder`, bare squares like `d5` become `square d5`. Without this, Bobcat returns `None` for most chess sentences. |

### `parse/`

| File | Purpose |
|------|---------|
| `ccg_parser.py` | Runs `preprocess.normalise()` on every sentence, filters by token length, feeds to Bobcat CCG parser, converts valid derivations to DisCoCat string diagrams, applies the IQP ansatz to compile them into parameterised quantum circuits. |

### `quantum/`

| File | Purpose |
|------|---------|
| `ansatz.py` | Stores `N_NOUN_QUBITS`, `N_SENTENCE_QUBITS`, `N_CLASSES`, `LABEL_MAP`. Exposes `build_ansatz()`. Every other quantum file imports constants from here — change qubit counts in one place. |
| `device.py` | Tries `lightning.gpu` → `lightning.qubit` → `default.qubit` in order and returns the fastest available PennyLane device. Also returns the correct differentiation method (`adjoint` for Lightning, `parameter-shift` for default). |
| `train.py` | Builds a `NumpyModel` from the compiled circuits, runs mini-batch SPSA training, logs loss and accuracy per epoch, returns the trained model and history dict. |
| `transfer.py` | Defines the style transfer QNode using the device from `device.py`. Trains U(φ) — a hardware-efficient 2-qubit unitary — to minimise infidelity between engine and GM sentence states. |
| `decode.py` | Builds a retrieval index over all GM sentences by storing their quantum probability vectors. At inference, applies U(φ) to an engine sentence state, converts to probabilities, and returns the top-k GM sentences by cosine similarity. This is what produces actual English text output. |
| `evaluate.py` | Prints per-class recall table. Saves `results/training.png` (loss + accuracy + infidelity curves) and `results/embeddings.png` (PCA of sentence quantum embeddings coloured by blunder class). |

---

## Prerequisites

### Python

Python **3.11** or **3.12**. Check with:

```bash
python --version
```

### Java (required for Bobcat CCG parser)

`lambeq` uses a Java-based parser under the hood.

**macOS**
```bash
brew install openjdk@17
```

**Ubuntu / Debian**
```bash
sudo apt install openjdk-17-jre-headless
```

**Windows** — download from [adoptium.net](https://adoptium.net), install, and add to PATH.

Verify:
```bash
java -version
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/darkisthenight07/QNLP-for-Chess-Annotation-Style-Transfer-and-Blunder-Pattern-Extraction.git
cd QNLP-for-Chess-Annotation-Style-Transfer-and-Blunder-Pattern-Extraction

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional — GPU acceleration for style transfer
pip install pennylane-lightning          # CPU C++ backend (always worth installing)
pip install pennylane-lightning[gpu]     # NVIDIA GPU backend (requires CUDA)

# 5. Download Bobcat model weights — one time, ~200 MB
python -c "from lambeq import BobcatParser; BobcatParser()"
```

---

## Adding Data

### Where to get PGN files

| Source | What to get | Link |
|--------|-------------|------|
| Lichess Open Database | Monthly exports with Stockfish evals | https://database.lichess.org |
| TWIC | Weekly GM tournament games | https://theweekinchess.com/twic |
| PGN Mentor | Annotated classic collections | https://www.pgnmentor.com |

### What format the PGN needs

The pipeline reads Stockfish evaluation comments in standard Lichess format:

```
1. e4 { [%eval 0.17] } e5 { [%eval 0.19] } 2. Nf3 { [%eval -1.50] } ...
```

Games without any `{ }` comment blocks are stored in the DB but produce no labeled sentences. Download Lichess exports with the **"with evaluations"** option enabled on their database page.

### Where to put your files

Drop any number of `.pgn` files directly into `pgn_files/`:

```
chess_qnlp/
└── pgn_files/
    ├── lichess_2024_january.pgn
    ├── kasparov_deep_blue.pgn
    └── twic_1500.pgn
```

The ingestion script reads all of them automatically.

**Recommended starting size:** 500–2000 annotated games. The Lichess monthly exports are very large — filter on their website for games rated 2000+ with evaluations before downloading.

---

## Running the Pipeline

### Full run

```bash
python main.py
```

Runs all 11 steps:

| Step | What happens |
|------|-------------|
| 1 | Ingest all PGN files from `pgn_files/` into SQLite |
| 2 | Assign blunder category labels to annotations |
| 3 | Load labeled sentences from DB (confidence ≥ 0.6) |
| 4 | Normalise chess notation, parse CCG, compile IQP circuits |
| 5 | Train blunder classifier with SPSA (120 epochs) |
| 6 | Print per-class recall, save embedding PCA plot |
| 7 | Train style transfer channel U(φ) (200 epochs) |
| 8 | Build GM retrieval index for the decoder |
| 9 | Save training curves to `results/training.png` |
| 10 | Run sample blunder classification, print predictions |
| 11 | Run sample style transfer, print top-3 GM sentence candidates |

### Running individual steps

```bash
# Ingest PGN files only
python -m data.ingest

# Label annotations only
python -m data.label

# Test the notation normaliser
python -m data.preprocess

# Parse a small sample and print circuits
python -m parse.ccg_parser
```

---

## Configuration

### `config.py`

```python
# Centipawn eval drop thresholds for blunder labeling
BLUNDER_THRESHOLDS = {
    "tactical_oversight":     -200,   # hung a piece, missed mate
    "positional_degradation":  -80,   # structural damage
    "time_pressure":           -50,   # moderate error under clock
    "opening_mistake":         -30,   # early concession
}

MIN_SENTENCE_TOKENS = 3    # shorter than this → filtered before CCG
MAX_SENTENCE_TOKENS = 12   # longer than this → too complex for IQP at 1–2 qubits
```

### `quantum/ansatz.py`

```python
N_NOUN_QUBITS     = 1   # qubits per noun word
N_SENTENCE_QUBITS = 2   # qubits for sentence register
N_LAYERS          = 1   # IQP circuit depth
N_CLASSES         = 4   # must equal 2 ** N_SENTENCE_QUBITS
```

If you increase `N_SENTENCE_QUBITS` to 3, set `N_CLASSES = 8` and add 4 more blunder categories to `LABEL_MAP`. Increasing `N_NOUN_QUBITS` to 2 gives richer word representations but doubles the circuit size and simulation time.

---

## Outputs

| File | Contents |
|------|---------|
| `chess_qnlp.db` | SQLite database with all games, moves, annotations, sentences, labels |
| `results/training.png` | Three plots: classifier cross-entropy loss, train/test accuracy, style transfer infidelity curve |
| `results/embeddings.png` | PCA of sentence quantum probability vectors, coloured by blunder class — shows whether the four types are separable in quantum embedding space |
| Terminal (step 10) | Per-sentence blunder type predictions with probabilities |
| Terminal (step 11) | For each engine-style sentence: top-3 GM-style candidate sentences with cosine similarity scores |

---

## Troubleshooting

**Most sentences return `None` from the parser**
This happened before `preprocess.py` was added. Make sure you are running the latest `ccg_parser.py` — it now normalises notation before parsing. If you still get low parse rates, check that your PGN comments contain actual English words and not just evaluation tags.

**`not enough data` error on startup**
Less than 10 labeled sentences found. Either no PGN files are in `pgn_files/`, the files have no `{ }` comments, or the eval format does not match `[%eval ...]`. Run `python -m data.ingest` separately and check its printed game count.

**Style transfer decoder prints nothing**
Fewer than 4 valid style pairs were found in the DB. This usually means the labeler did not produce enough `positional_degradation` labels. Lower the `BLUNDER_THRESHOLDS["positional_degradation"]` threshold in `config.py` (e.g. from -80 to -40) to label more sentences and retry `python -m data.label`.

**Training is very slow**
Install `pennylane-lightning` for a C++ CPU backend that is 5–10× faster than the default pure-Python simulator. For NVIDIA GPUs, `pennylane-lightning[gpu]` gives another 10–50× improvement. `quantum/device.py` detects and uses these automatically — no code changes needed.

**Java not found**
Install OpenJDK 17 as described in Prerequisites. The Bobcat CCG parser requires a JVM to run the underlying Java supertagger.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `lambeq` | ≥ 0.4.1 | DisCoCat framework — CCG parsing, string diagrams, IQP ansatz, NumpyModel, SPSA |
| `pennylane` | ≥ 0.36 | Style transfer QNode, parameter-shift and adjoint gradients, AdamOptimizer |
| `SQLAlchemy` | ≥ 2.0 | ORM layer over SQLite |
| `chess` | ≥ 1.10 | PGN file reading and game tree traversal |
| `scikit-learn` | ≥ 1.4 | PCA for embedding visualisation |
| `matplotlib` | ≥ 3.8 | Training curve and embedding plots |
| `numpy` | ≥ 1.26 | Numerical ops throughout |

Optional:

| Package | Purpose |
|---------|---------|
| `pennylane-lightning` | C++ CPU simulator, 5–10× faster than default |
| `pennylane-lightning[gpu]` | NVIDIA GPU simulator, 10–50× faster |
