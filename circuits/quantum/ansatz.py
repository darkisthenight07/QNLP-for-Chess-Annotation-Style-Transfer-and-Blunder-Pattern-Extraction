from lambeq import IQPAnsatz, AtomicType
from lambeq.backend.grammar import Diagram

N = AtomicType.NOUN
S = AtomicType.SENTENCE

N_NOUN_QUBITS     = 1
N_SENTENCE_QUBITS = 2
N_CLASSES         = 4
N_LAYERS          = 1

LABEL_MAP = {
    0: "tactical_oversight",
    1: "positional_degradation",
    2: "time_pressure",
    3: "opening_mistake",
}


def build_ansatz() -> IQPAnsatz:
    return IQPAnsatz(
        ob_map={N: N_NOUN_QUBITS, S: N_SENTENCE_QUBITS},
        n_layers=N_LAYERS,
        discard=False,
    )


def diagrams_to_circuits(diagrams: list[Diagram], ansatz: IQPAnsatz) -> list:
    return [ansatz(d) for d in diagrams]
