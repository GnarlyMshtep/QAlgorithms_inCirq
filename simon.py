"""Simons"""
import itertools
import cirq
from bitstring import BitArray, BitStream
from utils import add, multiply, U_f
import time

# n is the size input of the function (number of non ancilla bits)


def simon(f, n: int) -> BitArray:
    qbits, circuit = initSimons(f, n)
    constraints = []


def initSimons(f, n: int):
    """An attempt to initialize simons once and for all, making the qubits and the circuit only once"""
    # initialisze the qubits
    qbits = [cirq.LineQubit(i) for i in range(2*n)]

    # Create a circuit
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q)
                   for q in cirq.LineQubit.range(n))  # first round of Hs
    # add the Uf, first test the above
    u_f = U_f(f, n, n)
    circuit.append(u_f(*qbits))  # ?questionable syntax

    circuit.append(cirq.H(q)
                   for q in cirq.LineQubit.range(n))  # 2nd round of Hs

    circuit.append(cirq.measure(q, key=f"q{i}")
                   for i, q in enumerate(qbits[:n]))

    print("Circuit:")
    print(circuit)
    return qbits, circuit


def simonQuantumSubroutine(f, n: int, circuit: cirq.Circuit) -> BitArray:
    """simulate the circuit and return the s value which was returned"""

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=20)
    print("Results:")
    print(result)


if __name__ == "__main__":
    # start by constructing a function which meets the simon criterions
    s = 1
    n = 10

    def f(x):
        """where 0 \leq x< n and f meets the simon criterion on the f above"""
        return 0

    simon(f, 10)

"""
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

        print(f"average circuit time ({n=}): {total_time / (2 ** (n + 1))}")

        stop_time = time.perf_counter()
        print(f"total time {n=}: {stop_time - start_time}, cases = {num_cases}")
"""
