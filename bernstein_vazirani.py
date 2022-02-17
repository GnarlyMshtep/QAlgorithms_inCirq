"Bernstein-Vazirani"
import itertools
import time

import cirq
import numpy as np
<<<<<<< HEAD
from utils import add, multiply
=======
from utils import create_qubits
>>>>>>> fd9b9221c92824b3d46696fba8f084bfb8406c09



class U_f(cirq.Gate):
    def __init__(self, f, n):
        super(U_f, self)
        self._f = f
        self._n = n

    def _num_qubits_(self):
        return self._n + 1

    def _unitary_(self):
        side = 2 ** (self._n + 1)
        ret = np.zeros((side, side))

        for xb in range(2 ** (self._n + 1)):
            x = xb >> 1
            b = xb & 1
            y = self._f(x)
            ret[xb, x << 1 | (b ^ y)] = 1

        return ret

    def _circuit_diagram_info_(self, args):
        return ["U_f"] * (self._n + 1)
        # return tuple(f"U_f[{i}]" for i in range(self._n + 1))


def create_bv_circuit(qubits, u_f):
    # Initialize last qubit to 1
    yield cirq.X(qubits[-1])

    yield tuple(cirq.H(q) for q in qubits)
    yield u_f(*qubits)
    yield tuple(cirq.H(q) for q in qubits[:-1])

    yield tuple(cirq.measure(q, key=f"q{i}") for i, q in enumerate(qubits[:-1]))


def run(n, f):
    b = f(0)

    qubits = create_qubits(n + 1)

    U = U_f(f, n)
    # print("U_f")
    # for i in range(2 ** (n + 1)):
    #     print(U._unitary_()[i, :])

    circuit = cirq.Circuit()
    circuit.append(create_bv_circuit(qubits, U))

    # print("Circuit:")
    # print(circuit)

    simulator = cirq.Simulator()
    start_time = time.perf_counter()
    result = simulator.run(circuit)
    stop_time = time.perf_counter()

    # breakpoint()

    a = 0
    for i in range(n):
        [measurements_i] = result.measurements[f"q{i}"]
        if len(set(measurements_i)) != 1:
            raise ValueError("Something went wrong!")
        a = (a << 1) + measurements_i[0]

    # print("\t", a, b)
    return (a, b), (stop_time - start_time)


if __name__ == "__main__":
    max_n = 10
    max_cases = 1000

    for n in range(1, max_n + 1):
        start_time = time.perf_counter()

        cases = itertools.islice(
            ((ab >> 1, ab & 1) for ab in range(2 ** (n + 1))), 0, max_cases
        )

        total_time = 0
        num_cases = 0
        for a, b in cases:
            # print(a, b)
            (actual_a, actual_b), average_time = run(
                n, lambda x: add(multiply(a, x), b)
            )
            assert (actual_a, actual_b) == (a, b)
            total_time += average_time
            num_cases += 1

        print(f"average circuit time (n={n}): {total_time / (2 ** (n + 1))}")

        stop_time = time.perf_counter()
        print(
            f"total time n={n}: {stop_time - start_time}, cases = {num_cases}")
