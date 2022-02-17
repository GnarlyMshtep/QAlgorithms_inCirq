"""Simons"""

from distutils.log import error
from random import randint
import cirq
from bitstring import BitArray, BitStream
from numpy import average
import numpy as np
import sympy
from utils import add, multiply, U_f, result_to_bitlist, result_to_int, bitstring_to_int, subtruct_bitstrings, rref_f2
import time
import copy

# n is the size input of the function (number of non ancilla bits)

"""
when do I stop running and concede that 0^n is the soln?
"""


def simon(n: int, f) -> BitArray:
    constraints = []
    circuitRuntimes = []

    while len(constraints) < n-1:
        qbits, circuit = initSimons(f, n)

        start_time = time.perf_counter()
        bitstring_list, bitstring_int = runSimonQuantumSubroutine(
            circuit, qbits, n)
        print('got bitstring_list', bitstring_list)
        end_time = time.perf_counter()
        circuitRuntimes.append(end_time-start_time)
        # we don't care about 0 solns
        ret_matrix, _ = rref_f2(np.array(constraints + [bitstring_list]))
        if(bitstring_int != 0 and np.any(ret_matrix[-1])):
            constraints.append(bitstring_list)

    # we have n linearly independent constraints and can solve the matrix equation
    zeroVector = [0]*(n-1)
    # M = sympy.Matrix(constraints)
    # s, _ = M.gauss_jordan_solve(sympy.Matrix(zeroVector))
    # list_s = np.array(s).astype(np.float64).tolist()
    print('constraints', constraints)   
    _, sol = rref_f2(np.array(constraints))
    ret_s = bitstring_to_int(list(sol))
    if f(0) != f(ret_s):
        ret_s = 0

    return ret_s, sum(circuitRuntimes)


def is_linearly_indp(list_of_bitstrings_orig: list, new_bitstring: list):
    list_of_bitstrings = list_of_bitstrings_orig[:]
    list_of_bitstrings.append(new_bitstring)
    M = sympy.Matrix(list_of_bitstrings)
    M, _ = M.rref()
    lastRow = list(M[-1, :])
    zeroVec = [0] * len(new_bitstring)
    return (zeroVec != lastRow)


def is_linearly_indp_matan(list_of_bitstrings_orig: list, new_bitstring: list):
    """"
    do gausiann elimination

    find some vector with leftmost 1, use that to elminate all other vectors that have one in that poisiton
    dont use that row again, and look at columns further to the right
    """
    list_of_bitstrings = copy.deepcopy(list_of_bitstrings_orig)
    list_of_bitstrings.append(new_bitstring)
    n = len(list_of_bitstrings[0])
    counter = 0
    need_resorting = True

    while(counter < len(list_of_bitstrings)):
        # if we need to resort after doing some operations in the last step (or not re-sorting at all yet) do it!
        if(need_resorting):
            list_of_bitstrings.sort(
                key=lambda x: bitstring_to_int(x), reverse=True)
            need_resorting = False

        if(bitstring_to_int(list_of_bitstrings[len(list_of_bitstrings)-1]) == 0):
            return False

        # make sure that there is only one bitstring that has this column poisiton on
        if(list_of_bitstrings[counter][counter] == 1):
            need_resorting = True
            for i in range(len(list_of_bitstrings)):
                if(i != counter):
                    list_of_bitstrings[i] = subtruct_bitstrings(
                        list_of_bitstrings[i], list_of_bitstrings[counter])
        counter += 1

    # do a last check after the last subtruction
    if([0]*n in list_of_bitstrings):
        return False
    else:
        return True


def runSimonQuantumSubroutine(circuit: cirq.Circuit, qbits, n) -> BitArray:
    """simulate the circuit and return the s value which was returned"""

    simulator = cirq.Simulator()
    result = simulator.run(circuit)
    return result_to_bitlist(result, n), result_to_int(result, n)


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

    #print(circuit)
    return qbits, circuit


def constructSimonDict(n: int, s: int):
    rand_perm = np.random.permutation(2**n)
    simon_array = np.ones(2**n, dtype=int) - 2
    for i in range(2**n):
        if simon_array[i] == -1:
            simon_array[i] = simon_array[i ^ s] = rand_perm[i]
    return simon_array


if __name__ == "__main__":
    # start by constructing a function which meets the simon criterions
    MAX_S = 10
    MAX_N = 10

    # results = []

    circ_avg_runtimes = []
    tot_runtimes = []
    for n in range(2, MAX_N):
        runtime_per_n = []
        circ_runtime_per_n = []
        for s_itr in range(0, MAX_S):
            # make simon function
            s = randint(0, 2**n-1)

            simon_array = list(constructSimonDict(n, s))
            # build circuit make qbits

            start_time = time.perf_counter()  # start time

            ret_s, average_circuit_runtime_per_circ = simon(
                n, lambda x: simon_array[x])  # run simon routine

            if(ret_s != s):  # catch if simon failed
                print('error! ret_s:', ret_s, 's:',
                      s, 'simonDict', simon_array, n)
                exit(1)

            end_time = time.perf_counter()

            # add meassurement
            circ_runtime_per_n.append(average_circuit_runtime_per_circ)
            runtime_per_n.append(end_time-start_time)

        # add round messurements
        circ_avg_runtimes.append(circ_runtime_per_n)
        tot_runtimes.append(runtime_per_n)

    print(tot_runtimes, circ_avg_runtimes)
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
        print(
            f"total time {n=}: {stop_time - start_time}, cases = {num_cases}")
"""
