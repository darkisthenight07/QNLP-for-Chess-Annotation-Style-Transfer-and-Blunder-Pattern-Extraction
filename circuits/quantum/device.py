import pennylane as qml


def get_device(n_wires: int) -> qml.Device:
    for backend in ("lightning.gpu", "lightning.qubit", "default.qubit"):
        try:
            dev = qml.device(backend, wires=n_wires)
            if backend != "default.qubit":
                print(f"  quantum device: {backend} ({n_wires} wires)")
            return dev
        except (qml.DeviceError, ImportError, RuntimeError):
            continue
    raise RuntimeError("No suitable PennyLane device found.")


def get_diff_method(dev: qml.Device) -> str:
    name = dev.name if hasattr(dev, "name") else str(dev)
    if "lightning" in name.lower():
        return "adjoint"
    return "parameter-shift"
