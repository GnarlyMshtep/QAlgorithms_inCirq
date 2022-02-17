"""Simons"""
from itertools import chain, combinations
import itertools
from random import randint
import cirq
from bitstring import BitArray, BitStream
from numpy import subtract
import sympy
from utils import add, multiply, U_f, result_to_bitlist, result_to_int, bitstring_list_to_int, subtruct_bitstrings
import time
from numpy.linalg import solve, det
import copy

# n is the size input of the function (number of non ancilla bits)

"""
when do I stop running and concede that 0^n is the soln?
"""


def simon(f, n: int) -> BitArray:
    qbits, circuit = initSimons(f, n)
    constraints = []

    while len(constraints) < n-1:
        bitstring_list, bitstring_int = runSimonQuantumSubroutine(
            circuit, qbits, n)
        # we don't care about 0 solns
        if(bitstring_int != 0 and is_linearly_indp(constraints, bitstring_list)):
            constraints.append(bitstring_list)

    # we have n linearly independent constraints and can solve the matrix equation
    zeroVector = [0]*(n-1)
    M = sympy.Matrix(constraints)
    s = M.gauss_jordan_solve(sympy.Matrix(zeroVector))
    print(list(s))


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


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
                key=lambda x: bitstring_list_to_int(x), reverse=True)
            need_resorting = False

        if(bitstring_list_to_int(list_of_bitstrings[len(list_of_bitstrings)-1]) == 0):
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

    return qbits, circuit


def runSimonQuantumSubroutine(circuit: cirq.Circuit, qbits, n) -> BitArray:
    """simulate the circuit and return the s value which was returned"""

    simulator = cirq.Simulator()
    result = simulator.run(circuit)
    return result_to_bitlist(result, n), result_to_int(result, n)


def constructSimonDict(s: int, n: int):
    d = {}
    for i in range(2**n):
        r = randint(0, 2**n)
        if i not in d.keys():
            d[i] = r
            if (s ^ i) in d.keys():
                print("error!")
                exit(1)
            else:
                d[(s^i) % 2**n] = r

    return d


if __name__ == "__main__":
    # start by constructing a function which meets the simon criterions
    s = 5
    n = 4

    simonDict = constructSimonDict(s, n)
    print(simonDict)
    simon(lambda x: simonDict[x], n)

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
