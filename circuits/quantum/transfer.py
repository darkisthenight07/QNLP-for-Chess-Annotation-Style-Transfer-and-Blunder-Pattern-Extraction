import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from lambeq import NumpyModel

from quantum.device import get_device, get_diff_method

_dev         = get_device(n_wires=2)
_diff_method = get_diff_method(_dev)


def _build_circuit(dev, diff_method):
    @qml.qnode(dev, diff_method=diff_method)
    def circuit(engine_state: np.ndarray, phi: np.ndarray) -> np.ndarray:
        qml.StatePrep(engine_state, wires=[0, 1])
        for l in range(phi.shape[0]):
            qml.RY(phi[l, 0], wires=0)
            qml.RY(phi[l, 1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(phi[l, 2], wires=0)
            qml.RY(phi[l, 3], wires=1)
        return qml.state()
    return circuit


_transfer_circuit = _build_circuit(_dev, _diff_method)


def extract_state(circuit, model: NumpyModel) -> np.ndarray:
    probs      = model([circuit])[0]
    amplitudes = np.sqrt(np.maximum(probs, 0))
    norm       = np.linalg.norm(amplitudes) + 1e-9
    return (amplitudes / norm).astype(complex)


def _infidelity(phi, engine_states, gm_states):
    total = pnp.zeros(1)
    for e_state, g_state in zip(engine_states, gm_states):
        out     = _transfer_circuit(e_state, phi)
        overlap = pnp.abs(pnp.sum(pnp.conj(pnp.array(g_state)) * out)) ** 2
        total   = total + overlap
    return 1.0 - total / len(engine_states)


def train_transfer(
    engine_circuits: list,
    gm_circuits: list,
    model: NumpyModel,
    n_layers: int = 3,
    n_epochs: int = 200,
    lr: float = 0.05,
) -> tuple[pnp.ndarray, dict]:

    engine_states = [extract_state(c, model) for c in engine_circuits]
    gm_states     = [extract_state(c, model) for c in gm_circuits]

    phi       = pnp.random.uniform(0, 2 * np.pi, (n_layers, 4), requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=lr)
    history   = {"infidelity": []}

    for epoch in range(n_epochs):
        phi, loss = optimizer.step_and_cost(
            lambda p: _infidelity(p, engine_states, gm_states), phi
        )
        history["infidelity"].append(float(loss))
        if epoch % 40 == 0:
            print(f"  epoch {epoch:3d} | infidelity {float(loss):.4f} | "
                  f"fidelity {1 - float(loss):.4f}")

    return phi, history


def apply_transfer(engine_circuit, model: NumpyModel, phi: pnp.ndarray) -> np.ndarray:
    e_state = extract_state(engine_circuit, model)
    return np.array(_transfer_circuit(e_state, phi))
