"Deutsch-Jozsa"
import enum
import itertools
import time

import cirq
import numpy as np


class Outcome(enum.Enum):
    CONSTANT = enum.auto()
    BALANCED = enum.auto()


def create_qubits(n):
    return [cirq.GridQubit(i, 0) for i in range(n)]


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


def create_dj_circuit(qubits, u_f):
    # Initialize last qubit to 1
    yield cirq.X(qubits[-1])

    yield tuple(cirq.H(q) for q in qubits)
    yield u_f(*qubits)
    yield tuple(cirq.H(q) for q in qubits[:-1])

    yield tuple(cirq.measure(q, key=f"q{i}") for i, q in enumerate(qubits[:-1]))


def run(n, f):
    qubits = create_qubits(n + 1)

    U = U_f(f, n)
    # print("U_f")
    # print(U._unitary_())

    circuit = cirq.Circuit()
    circuit.append(create_dj_circuit(qubits, U))

    # print("Circuit:")
    # print(circuit)

    simulator = cirq.Simulator()
    start_time = time.perf_counter()
    result = simulator.run(circuit)
    stop_time = time.perf_counter()

    # print(result)

    ones_count = np.sum(([v for v in result.measurements.values()]))
    if ones_count == 0:  # No 1's, so all 0's
        return Outcome.CONSTANT, (stop_time - start_time)
    else:  # Found some 1's
        return Outcome.BALANCED, (stop_time - start_time)


if __name__ == "__main__":
    max_n = 10
    max_balanced_cases = 998  # 998 + 2 constant cases = 1000 max cases

    for n in range(1, max_n + 1):
        start_time = time.perf_counter()

        constant_outputs = [[False] * (2**n), [True] * (2**n)]
        # balanced_outputs = ([bool(i & (1 << j)) for j in range(n+1, -1, -1)] for i in range(2 ** (n+1)))
        balanced_outputs = list(
            itertools.islice(
                itertools.permutations(
                    [False] * (2 ** (n - 1)) + [True] * (2 ** (n - 1))
                ),
                0,
                max_balanced_cases,
            )
        )

        constant_function_cases = (
            (lambda x: co[x], Outcome.CONSTANT) for co in constant_outputs
        )
        balanced_function_cases = (
            (lambda x: bo[x], Outcome.BALANCED) for bo in balanced_outputs
        )

        # for f, expected in [
        #     (lambda _: False, Outcome.CONSTANT),
        #     (lambda _: True, Outcome.CONSTANT),
        #     (lambda x: bool(x & 1), Outcome.BALANCED),
        #     (lambda x: not bool(x & 1), Outcome.BALANCED),
        # ]:
        total_time = 0
        for f, expected in itertools.chain(
            constant_function_cases, balanced_function_cases
        ):
            # print(expected)
            output, average_time = run(n, f)
            assert output == expected
            total_time += average_time

        print(
            f"average circuit time ({n=}): {total_time / (len(constant_outputs) + len(balanced_outputs))}"
        )

        stop_time = time.perf_counter()
        print(
            f"total time {n=}: {stop_time - start_time}, cases = {(len(constant_outputs) + len(balanced_outputs))}"
        )
