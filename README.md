# QNLP for Chess Annotation: Style Transfer and Blunder Pattern Extraction

The long-term goal of this project is to see whether Quantum NLP (using `lambeq`)
can be used to model chess commentary — both for classifying *why* a blunder
happened (tactical oversight, positional decay, time pressure, opening
mistake) and eventually for generating grandmaster-style commentary from raw
board states.

Right now this repo is still early. The data pipeline and the classical
label-generation step are working end to end. The CCG parsing / quantum
circuit compilation step is wired up but hasn't been run against a real
corpus yet. Nothing about the actual quantum classifier or the text
generation side (Phase 3/4 in my notes) is implemented — that's still just
a plan.

## What's actually here

**`data/ingest.py`**
Reads PGN files (with Stockfish `[%eval ...]` comments, like the ones you get
from the Lichess database or annotated GM games) and pulls out, per move:
the SAN move text, the engine eval before/after, and any human comment
attached to that move. It stores all of this in a SQLite database via
SQLAlchemy. Comments get cleaned of PGN clock/eval tags and split into short
sentences (3–12 tokens) so they're small enough to eventually feed into a
CCG parser.

**`data/label.py`**
Takes every annotation in the DB and tries to assign it a blunder category.
It does this two ways and reconciles them: a keyword-matching pass (looks
for words like "fork", "time trouble", "isolated pawn", etc.) and a
threshold pass on the eval drop (e.g. anything worse than -200 centipawns
gets called a tactical oversight). If both methods agree, confidence is
high; if they disagree, the eval-based label wins since it's grounded in
something objective; if only one method fires, that one is used with lower
confidence.

**`parse/ccg_parser.py`**
Uses `lambeq`'s `BobcatParser` to turn a sentence into a CCG string diagram,
then compiles that diagram into a parameterised quantum circuit using an IQP
ansatz. This is the piece that's supposed to turn "White missed a fork on
the kingside" into something a quantum classifier could eventually train
on. The code runs, but I haven't validated it against a large batch of real
annotations yet — treat it as a working prototype, not a finished pipeline.

**`models.py` / `config.py`**
ORM schema (games → moves → annotations → corpus_sentences /
blunder_labels) and shared config (DB path, blunder thresholds, sentence
length bounds).

## What's NOT here yet

- No actual quantum training loop. No classifier sitting on top of the
  circuits, no optimizer, no loss function.
- No RAG / generation pipeline for producing new commentary in a GM's
  style.
- No PGN files are included in the repo — you have to supply your own.

## Setup

You'll need Python 3.10+ (the code uses `X | Y` style type hints, so
anything older will break on import).

```bash
git clone https://github.com/darkisthenight07/QNLP-for-Chess-Annotation-Style-Transfer-and-Blunder-Pattern-Extraction.git
cd QNLP-for-Chess-Annotation-Style-Transfer-and-Blunder-Pattern-Extraction
pip install -r requirements.txt
```

Heads up, `lambeq` pulls in a fair number of dependencies (it needs a
CCG parser model and PyTorch under the hood), so the install can take a
few minutes.

## How to run what's currently built

### 1. Get some annotated PGN files

You need PGN files that contain both engine evaluations and text comments,
for example something exported from lichess with analysis, or an annotated
game collection. Drop them into a `pgn_files/` folder in the project root
(the ingestion script will create this folder for you if it doesn't
exist, but it won't be populated automatically — you have to add the
files yourself).

### 2. Ingest the games into the database

```bash
python -m data.ingest
```

This walks every `.pgn` file in `pgn_files/`, parses each game move by
move, and stores games, moves, annotations, and split sentences into a
SQLite file called `chess_qnlp.db` in the project root. You'll see a
per-file count of games processed printed to the console.

### 3. Generate blunder labels

```bash
python -m data.label
```

This reads every annotation currently in the database and tries to tag it
with a blunder category based on keywords and eval drop. At the end it
prints how many annotations got labeled vs skipped (skipped usually means
there wasn't enough signal from either the keywords or the eval drop to
confidently pick a category), plus a breakdown of how many annotations
fell into each category.

### 4. Parse sentences into quantum circuits (experimental)

```bash
python -m parse.ccg_parser
```

This pulls the sentences generated in step 2 out of the database, runs
them through the CCG parser, and compiles each resulting diagram into an
IQP circuit. By default it only processes the first 50 sentences (this is
hardcoded in the `__main__` block right now, change the `limit` argument in
`run_parsing()` if you want to try more). It prints how many sentences were
loaded, how many parsed successfully, how many circuits got compiled, and
shows a handful of example sentences that made it through. Some sentences
will fail to parse — that's expected, CCG parsers can be picky about
punctuation and unusual chess phrasing.

## A note on the database

The DB path and blunder thresholds live in `config.py`. If you want to
start fresh, just delete `chess_qnlp.db` and rerun the ingestion step —
the schema will be recreated automatically.

## Where this is headed

The next real milestone is hooking the compiled quantum circuits from
`ccg_parser.py` up to an actual classifier (probably starting with
Pauli-Z expectation values fed into a small logistic regression or shallow
net) and training it against the blunder labels from `data/label.py`. After
that, the harder problem — generating GM-style commentary from a board
state using the trained circuit as a semantic encoder plus a RAG layer —
is still mostly an idea on paper at this point.
