import random
import time
from math import pi, acos, floor, sin
from utils import U_f, create_qubits
import cirq
import sys


def create_grover_iterated_subcircuit(qubits, u_f, k):
    ''' 
    Grover subcircuit for the quantum step
    k = number of Grover iterations
    '''
    n = u_f.nInp
    # assume num qubits = n + 1
    # initialize last qubit to 1
    yield cirq.X(qubits[-1])
    # input = |++....++->
    yield tuple(cirq.H(q) for q in qubits)
    # Grover section
    for i in range(k):
        # Z_f
        yield u_f(*qubits)
        # H^n
        yield tuple(cirq.H(q) for q in qubits[:-1])
        # Z_0
        yield cirq.X.controlled(n, [0]*n).on(*qubits)
        # H^n
        yield tuple(cirq.H(q) for q in qubits[:-1])

    yield tuple(cirq.measure(q, key=f"q{i}") for i, q in enumerate(qubits[:-1]))


def calc_grover_params(a, n):
    N = 1 << n
    # the angle of the grover circuit as a rotation matrix in the special subspace
    theta = acos(1-2*a/N) / 2
    # we choose the integer value of k which minimizes |pi/2 - (2k+1)*theta|
    # floor(x + 0.5) = round(x)
    k = floor(pi/(4*theta) + 0.5) - 1
    final_sin = sin((2*k+1)*theta)
    success_prob = final_sin ** 2
    return k, success_prob


def grover(qubits, u_f, a):
    n = u_f.nInp
    assert u_f.nAnc == 1
    k, _ = calc_grover_params(a, n)
    return create_grover_iterated_subcircuit(qubits, u_f, k)


def run_grover(f, n, a):
    qubits = create_qubits(n+1)
    u_f = U_f(f, n, 1)
    circuit = cirq.Circuit()
    circuit.append(grover(qubits, u_f, a))
    start_time = time.perf_counter()
    simulator = cirq.Simulator()
    stop_time = time.perf_counter()
    result = simulator.run(circuit)
    result_int = sum(result.measurements[f'q{i}'][0][0] << (
        n-i-1) for i in range(n))
    return (result_int, stop_time - start_time)


if __name__ == "__main__":
    num_bits = None
    num_cases = None
    if len(sys.argv) == 3:
        num_bits, num_cases = tuple(map(int, sys.argv[1:]))
    else:
        print("Usage: ./grover.py NUM_BITS NUM_CASES")
        exit(1)

    total_run_time = 0
    num_correct = 0
    k, success_prob = calc_grover_params(1, num_bits)
    for i in range(num_cases):
        N = 1 << num_bits
        r = random.randint(0, N)
        print(f"Trial: expected answer {r:b}")
        def f(x): return int(x == r)
        result, run_time = run_grover(f, num_bits, 1)
        print(f"Got answer: {result:b}")
        num_correct += f(result)
        total_run_time += run_time
    print("")
    print(f"Using k={k} iterations")
    print(
        f"Expected correct: {floor(success_prob*num_cases+0.5)}/{num_cases}")
    print(f"Total correct: {num_correct}/{num_cases}")
    print(f"Average run time: {total_run_time/num_cases:3f} s")
