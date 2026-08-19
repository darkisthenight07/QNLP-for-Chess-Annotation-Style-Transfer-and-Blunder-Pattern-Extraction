# QNLP for Chess Annotation Style Transfer and Blunder Pattern Extraction

A quantum natural language processing pipeline that classifies chess blunder types from grandmaster commentary and learns to transfer annotation style between engine output and human GM language.

Built using the **DisCoCat** (Distributional Compositional Categorical) framework via `lambeq`, with quantum simulation via `PennyLane`.

---

## What This Project Does

Chess annotations like *"White misses the fork on d5"* have compositional grammatical structure that mirrors the logical structure of the chessboard. Classical NLP ignores this structure. This project instead:

1. **Parses** each annotation sentence through a CCG grammar parser (Bobcat) to produce a string diagram — a wiring blueprint where each word is a typed morphism.
2. **Compiles** that diagram into a parameterised quantum circuit using the IQP (Instantaneous Quantum Polynomial) ansatz, where nouns become qubit states and grammatical reductions (subject-verb-object contractions) become entangling gates.
3. **Trains** the circuit parameters end-to-end to classify annotations into four blunder categories: tactical oversight, positional degradation, time pressure, and opening mistake.
4. **Learns a style transfer channel** — a small unitary U(φ) on the sentence qubit register — that rotates engine-style sentence embeddings toward grandmaster-style embeddings.

---

## Project Status

This is a working prototype. The pipeline is complete end-to-end. What it does well:

- Full PGN ingestion and structured SQLite storage
- Eval-drop + keyword labeling with confidence scores
- CCG parsing via lambeq's Bobcat parser (pre-trained neural supertagger)
- IQP circuit compilation from grammatical string diagrams
- SPSA-based quantum circuit training (hardware-safe, no backprop through circuits)
- Quantum channel learning for style transfer via PennyLane parameter-shift gradients

What is limited by current qubit counts:

- Sentence circuits are small (2–6 qubits depending on sentence length). This is fine for simulation but constrains expressivity.
- Style transfer decodes quantum states back to text via nearest-neighbour lookup, not generation. True quantum text generation requires far more qubits than current simulators handle practically.
- Training on the full Lichess corpus would require moving from `NumpyModel` (exact simulation) to `TketModel` with a real backend (IBM, IonQ), or a GPU-accelerated simulator.

---

## Prerequisites

### Python version

Python **3.11** or **3.12**. Do not use 3.13 — `lambeq` and `pennylane` have not fully validated on it yet.

Check yours:

```bash
python --version
```

### System dependencies

`lambeq` uses a Java-based CCG parser (EasyCCG/Bobcat) under the hood.

**macOS:**
```bash
brew install openjdk@17
```

**Ubuntu / Debian:**
```bash
sudo apt install openjdk-17-jre-headless
```

**Windows:** Download and install [OpenJDK 17](https://adoptium.net/). Add it to your PATH.

Verify Java is available:
```bash
java -version
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/darkisthenight07/QNLP-for-Chess-Annotation-Style-Transfer-and-Blunder-Pattern-Extraction.git
cd QNLP-for-Chess-Annotation-Style-Transfer-and-Blunder-Pattern-Extraction

# 2. Create a virtual environment (strongly recommended)
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. lambeq downloads its Bobcat model weights on first use (~200 MB)
#    Run this once to trigger the download before running the pipeline:
python -c "from lambeq import BobcatParser; BobcatParser()"
```

---

## File Structure

```
chess_qnlp/
│
├── README.md
├── requirements.txt
├── config.py                  ← paths, DB URL, blunder thresholds
├── models.py                  ← SQLAlchemy ORM (5 tables)
├── main.py                    ← full pipeline entry point
│
├── pgn_files/                 ← PUT YOUR PGN FILES HERE (created on first run)
├── results/                   ← plots saved here (created on first run)
├── chess_qnlp.db              ← SQLite database (created on first run)
│
├── data/
│   ├── ingest.py              ← PGN parsing → SQLite
│   └── label.py               ← eval drop + keywords → blunder label
│
├── parse/
│   └── ccg_parser.py          ← Bobcat CCG → string diagram → IQP circuit
│
└── quantum/
    ├── ansatz.py              ← ansatz config, qubit counts, label map
    ├── train.py               ← NumpyModel + SPSA training loop
    ├── transfer.py            ← PennyLane style transfer channel
    └── evaluate.py            ← accuracy report, PCA plot, training curves
```

---

## Adding Data

### Where to get PGN files

The project is built around annotated games — PGNs where moves have `{ comment }` blocks with Stockfish evaluations or human annotations.

**Best free sources:**

| Source | What to download | URL |
|--------|-----------------|-----|
| Lichess Open Database | Monthly PGN exports with Stockfish evals | https://database.lichess.org |
| TWIC (This Week in Chess) | GM tournament games, often annotated | https://theweekinchess.com/twic |
| PGN Mentor | Classic annotated game collections | https://www.pgnmentor.com |

### What format the PGN needs to have

The ingestion script looks for Stockfish evaluation comments in this format inside `{ }` blocks:

```
1. e4 { [%eval 0.17] } e5 { [%eval 0.25] } 2. Nf3 { [%eval -1.50] [%clk 0:05:00] } ...
```

Lichess exports with evaluations turned on use exactly this format. Games without any `{ }` comments will be stored but produce no sentences or labels.

### How to add your files

```
chess_qnlp/
└── pgn_files/
    ├── lichess_2024_01.pgn
    ├── kasparov_annotated.pgn
    └── twic_1500.pgn
```

Just drop any number of `.pgn` files into the `pgn_files/` folder. The ingestion script reads all of them automatically.

**Recommended starting size:** 500–2000 annotated games gives enough labeled sentences to train meaningfully. The full Lichess monthly exports are millions of games — start with a filtered subset (e.g. games rated 2000+ with evals enabled, which you can filter on the Lichess database page before downloading).

---

## Running the Pipeline

### Full pipeline (recommended first run)

```bash
python main.py
```

This runs all phases in order:
1. Ingests all PGN files from `pgn_files/` into SQLite
2. Labels annotations with blunder categories
3. Loads labeled sentences, parses them with Bobcat CCG
4. Compiles IQP quantum circuits
5. Trains the blunder classifier (120 epochs, SPSA)
6. Evaluates and saves plots to `results/`
7. Trains the style transfer channel (200 epochs)
8. Runs inference on four sample sentences

### Running individual phases

```bash
# Phase 1a: ingest PGN files only
python -m data.ingest

# Phase 1b: label the annotations
python -m data.label

# Phase 2: parse sentences and print sample circuits
python -m parse.ccg_parser
```

### Expected runtime

| Phase | What happens | Typical time |
|-------|-------------|--------------|
| Ingestion | PGN parsing, DB writes | ~1 min per 1000 games |
| Labeling | SQL queries + keyword scan | <30 seconds |
| CCG parsing | Bobcat neural supertagger | ~2–5 sec per 100 sentences |
| Circuit compilation | IQP ansatz application | ~1 sec per 100 circuits |
| Classifier training | 120 epochs, NumpyModel sim | ~5–20 min (CPU) |
| Style transfer | 200 epochs, PennyLane | ~10–30 min (CPU) |

---

## Configuration

All tuneable values are in `config.py`:

```python
# Blunder category centipawn thresholds (eval drop in centipawns)
BLUNDER_THRESHOLDS = {
    "tactical_oversight":     -200,   # massive blunder, hung a piece
    "positional_degradation":  -80,   # structural damage
    "time_pressure":           -50,   # moderate error
    "opening_mistake":         -30,   # early concession
}

# Sentence length filter for CCG parsing
MIN_SENTENCE_TOKENS = 3
MAX_SENTENCE_TOKENS = 12
```

To change qubit counts or circuit depth, edit `quantum/ansatz.py`:

```python
N_NOUN_QUBITS     = 1   # qubits per noun word
N_SENTENCE_QUBITS = 2   # qubits for sentence register (2^2 = 4 classes)
N_LAYERS          = 1   # IQP circuit depth
```

If you change `N_SENTENCE_QUBITS`, also update `N_CLASSES` to `2 ** N_SENTENCE_QUBITS`.

---

## Outputs

After a full run, `results/` will contain:

| File | Contents |
|------|----------|
| `results/training.png` | Classifier loss, train/test accuracy, transfer infidelity curves |
| `results/embeddings.png` | PCA of sentence qubit distributions, coloured by blunder class |

The terminal prints per-class recall and sample predictions.

---

## Common Errors

**`lambeq` fails to parse many sentences → returns `None`**
Normal for sentences that are too short, too long, or contain chess notation the CCG tagger has not seen (e.g. bare move notations like `Nf3`). The pipeline filters these out. Increase your dataset size to compensate.

**`Java not found` error on first run**
Install OpenJDK 17 as described in Prerequisites. The Bobcat parser requires a JVM.

**`NumpyModel` training is very slow**
This is exact statevector simulation. On CPU it is O(2^n_qubits) per forward pass. Keep `N_NOUN_QUBITS = 1` and `N_SENTENCE_QUBITS = 2` unless you have significant compute. For GPU acceleration, switch to `pennylane-lightning` (`pip install pennylane-lightning`) and change the device in `quantum/transfer.py` to `"lightning.gpu"`.

**`not enough data` error in `main.py`**
Less than 10 labeled sentences were found. This means either: no PGN files were added, the PGN files have no `{ }` comments, or the eval format does not match `[%eval ...]`. Check your PGN source.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `lambeq` | DisCoCat framework — CCG parsing, string diagrams, IQP ansatz, NumpyModel |
| `pennylane` | Quantum circuit simulation for style transfer, parameter-shift gradients |
| `SQLAlchemy` | ORM layer over SQLite for structured data storage |
| `chess` | PGN file reading and game tree traversal |
| `scikit-learn` | PCA for embedding visualisation |
| `matplotlib` | Training curve and embedding plots |
| `numpy` | Numerical ops throughout |
