import numpy as np
from lambeq import NumpyModel

from quantum.transfer import extract_state, apply_transfer
from pennylane import numpy as pnp


class StyleDecoder:
    """
    Converts a transferred quantum state back into natural language
    by nearest-neighbour retrieval over a pre-built GM sentence index.

    How it works
    ------------
    1. At build time: run every GM sentence circuit through the trained
       QNLP model to get its 4-dim probability vector (the embedding).
       Store all vectors + their source sentences in a matrix.

    2. At decode time: apply the transfer unitary U(phi) to the engine
       sentence state. The output is a 4-dim complex amplitude vector.
       Compute its probability vector (|amplitude|^2). Find the GM
       sentence whose stored embedding has the highest cosine similarity
       to this probability vector. Return that sentence.

    Why this works
    --------------
    The QNLP model maps sentences to points in a 4-dim probability
    simplex. Similar sentences (same blunder type, same grammatical
    structure) cluster together in this space. After training, U(phi)
    rotates engine-style points toward the GM cluster. Nearest-neighbour
    retrieval finds the actual GM sentence that lives closest to that
    rotated point, giving us a real English output.
    """

    def __init__(self) -> None:
        self.gm_sentences:  list[str]   = []
        self.gm_embeddings: np.ndarray  = None

    def build_index(
        self,
        gm_sentences: list[str],
        gm_circuits: list,
        model: NumpyModel,
    ) -> None:
        embeddings = []
        for circuit in gm_circuits:
            probs = model([circuit])[0]
            probs = np.maximum(probs, 0)
            probs = probs / (probs.sum() + 1e-9)
            embeddings.append(probs)

        self.gm_sentences  = gm_sentences
        self.gm_embeddings = np.array(embeddings)
        print(f"  decoder index built: {len(gm_sentences)} GM sentences")

    def _prob_vector(self, state: np.ndarray) -> np.ndarray:
        probs = np.abs(state) ** 2
        probs = np.maximum(probs, 0)
        return probs / (probs.sum() + 1e-9)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        norm_a = np.linalg.norm(a) + 1e-9
        norm_b = np.linalg.norm(b, axis=1) + 1e-9
        return (b @ a) / (norm_b * norm_a)

    def decode(
        self,
        engine_circuit,
        model: NumpyModel,
        phi: pnp.ndarray,
        top_k: int = 3,
    ) -> list[dict]:
        if self.gm_embeddings is None:
            raise RuntimeError("Call build_index() before decode().")

        transferred_state = apply_transfer(engine_circuit, model, phi)
        query_probs       = self._prob_vector(transferred_state)
        similarities      = self._cosine_similarity(query_probs, self.gm_embeddings)
        ranked            = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "sentence":   self.gm_sentences[i],
                "similarity": float(similarities[i]),
                "rank":       rank + 1,
            }
            for rank, i in enumerate(ranked)
        ]

    def decode_batch(
        self,
        engine_circuits: list,
        model: NumpyModel,
        phi: pnp.ndarray,
        top_k: int = 1,
    ) -> list[list[dict]]:
        return [self.decode(c, model, phi, top_k) for c in engine_circuits]


def build_decoder(
    gm_sentences: list[str],
    gm_circuits: list,
    model: NumpyModel,
) -> StyleDecoder:
    decoder = StyleDecoder()
    decoder.build_index(gm_sentences, gm_circuits, model)
    return decoder
