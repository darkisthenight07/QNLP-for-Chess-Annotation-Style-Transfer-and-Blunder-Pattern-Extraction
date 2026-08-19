import re


_PIECE_NAMES = {
    "K": "king",
    "Q": "queen",
    "R": "rook",
    "B": "bishop",
    "N": "knight",
}

_SQUARE_RE   = re.compile(r"\b[a-h][1-8]\b")
_SAN_RE      = re.compile(r"\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b")
_EVAL_TAG_RE = re.compile(r"\[%[^\]]+\]")
_MULTI_WS_RE = re.compile(r"\s+")


def _expand_san(match: re.Match) -> str:
    san = match.group(1)
    if not san or len(san) < 2:
        return san
    piece_letter = san[0] if san[0] in _PIECE_NAMES else ""
    piece_word   = _PIECE_NAMES.get(piece_letter, "pawn")
    square       = san[-2:].rstrip("+#") if len(san) >= 2 else ""
    captured     = "captures on" if "x" in san else "moves to"
    if piece_letter:
        return f"{piece_word} {captured} {square}"
    return f"pawn {captured} {square}"


def _expand_squares(text: str) -> str:
    file_names = {"a": "a", "b": "b", "c": "c", "d": "d",
                  "e": "e", "f": "f", "g": "g", "h": "h"}
    def _sq(m: re.Match) -> str:
        sq = m.group(0)
        return f"square {file_names[sq[0]]}{sq[1]}"
    return _SQUARE_RE.sub(_sq, text)


def normalise(text: str) -> str:
    text = _EVAL_TAG_RE.sub("", text)
    text = _SAN_RE.sub(_expand_san, text)
    text = text.replace("!!", " strong move.").replace("??", " blunder.")
    text = text.replace("!?", " interesting move.").replace("?!", " dubious move.")
    text = text.replace("!", ".").replace("?", ".")
    text = _expand_squares(text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def normalise_batch(sentences: list[str]) -> list[str]:
    return [normalise(s) for s in sentences]


if __name__ == "__main__":
    examples = [
        "Nf3! exploits the pin on the c-file [%eval -1.50]",
        "White plays Bxf7+ and wins the exchange on e8",
        "The rook on d1 controls the d-file completely",
        "?? allows Qd8# immediately",
    ]
    for raw in examples:
        print(f"  raw  : {raw}")
        print(f"  clean: {normalise(raw)}")
        print()
