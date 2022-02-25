# Shor's algorithm using Cirq
# Based on Scott Aaronson's notes
# https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-845-quantum-complexity-theory-fall-2010/lecture-notes/MIT6_845F10_lec07.pdf
# Cost: < 3*n qubits where n is the number of bits encoding the number to be factored

from cirq import sim
from cirq.circuits import circuit
import time
import random
import sys
from utils import U_f, create_qubits
import cirq
from math import log2, ceil, floor

# Use new randomness every time
random.seed(time.time())


def modexp(mod, base, e):
    n = ceil(log2(e+1))
    ret = 1
    for i in range(n):
        ret = (ret * ret) % mod
        if (e >> (n-i-1)) & 1:
            ret = (ret * base) % mod
    return ret


def gcd(a, b):
    '''
    Euclidean algorithm
    '''
    if a == 0:
        return b
    return gcd(b % a, a)


def is_prime(N):
    '''
    A simple, okay-ish primality test
    Which returns True for 1 also (ok for our purposes)
    '''
    for i in range(2, int(N**1/2 + 2)):
        if N % i == 0:
            return False
    return True


def is_power(N):
    for i in range(2, int(log2(N))+4):
        if N == floor(N ** (1/i) + 0.5)**i:
            return True, i
    return False, 1


EPS = 1e-9


def extract_denominator(x, N):
    '''
    Compute continued fraction representation of (floating point) x
    Then return the last denominator in the sequence of approximations
    Which is still less than N 
    '''
    c_frac_repr = []
    prev_den = 1
    curr_den = 0
    while True:
        flx = floor(x)
        c_frac_repr.append(flx)
        # print(c_frac_repr)
        prev_den, curr_den = curr_den, flx * curr_den + prev_den
        # print(prev_den, curr_den)
        if curr_den > N:
            return prev_den
        xm1 = x % 1
        if xm1 < EPS:
            return curr_den
        x = 1. / xm1


def create_period_finder(qubits, u_f):
    r1_length = u_f.nInp
    r2_length = u_f.nAnc
    # initialize to |++++++++>|0000>
    yield tuple(cirq.H(qubits[i]) for i in range(r1_length))
    # apply u_f
    yield u_f(*qubits)
    # measure second half qubits
    yield tuple(cirq.measure(q, key=f"q{i+r1_length}") for i, q in enumerate(qubits[r1_length:]))
    # apply QFT
    yield cirq.qft(*(qubits[:r1_length]))
    # measure
    yield tuple(cirq.measure(q, key=f"q{i}") for i, q in enumerate(qubits[:r1_length]))


shrink_factors = [0, 0, 0, 0, 0, 1, 4]


def run_period_finder(N, x):
    print(f"Running period finder: trying {x}")
    n = ceil(log2(N))

    def f(r):
        return modexp(N, x, r)
    r1_length = 2*n - shrink_factors[n]
    r2_length = n
    print(
        f"Using {r1_length} qubit input register and {r2_length} ancillary qubits")
    qubits = create_qubits(r1_length + r2_length)
    u_f = U_f(f, r1_length, r2_length)
    x_time = 0
    x_invocations = 0
    while True:
        circuit = cirq.Circuit()
        circuit.append(create_period_finder(qubits, u_f))

        simulator = cirq.Simulator()
        # simulator = qsimcirq.QSimSimulator()
        print("--SIMULATING--")
        start_time = time.perf_counter()
        # result = simulator.compute_amplitudes(
        # circuit, bitstrings=[i for i in range(256)])
        result = simulator.run(circuit)
        stop_time = time.perf_counter()

        x_time += stop_time - start_time
        x_invocations += 1

        result_int = sum(result.measurements[f'q{i}'][0][0] << (
            r1_length-i-1) for i in range(r1_length))
        # result ~= k * Q /s
        # where k is some integer, s is desired answer
        # result / Q ~= k / s
        Q = 1 << r1_length
        print(f"Got QFT result: {result_int}/{Q}")
        # if N has multiple prime factors, no element has order greater than (N-1)/2
        guess = extract_denominator(result_int / Q, N // 2 + 1)
        # Seems unlikely to get a number bigger than s
        print(f"Guessing denominator {guess}")
        if modexp(N, x, guess) == 1:
            return guess, x_time, x_invocations


def run_shor(N, extend=False):
    '''
    Following the discussion on Piazza question 213, we extend Shor to 
    also use exponents divisible by 3.
    '''
    tot_time = 0.
    tot_invocations = 0.
    factor = 1
    if is_prime(N):
        print(f"Boo, {N} is prime or 1!")
        return
    is_pwr, pwr = is_power(N)
    if is_pwr:
        print(f"Boo, {N} is a {pwr}nd/rd/th power")
        return
    if N % 2 == 0:
        print(f"Boo, {N} is even!")
        return
    while True:
        x = random.randint(1, N-1)
        g = gcd(x, N)
        if g > 1:
            # if we were actually trying to factor, this would be success
            # but in the interest of demonstrating Shor, we continue
            continue
        r, x_time, x_invocations = run_period_finder(N, x)
        tot_time += x_time
        tot_invocations += x_invocations
        factor = 1
        if r % 2 == 0:
            # abusing python big ints here
            # x ** r - 1 mod N can be computed efficiently
            factor = gcd(modexp(N, x, r//2) - 1, N)
        elif extend and r % 3 == 0:
            factor = gcd(modexp(N, x, r//3) - 1, N)
        if factor == 1:
            print(f"Darn, (x,r)=({x},{r}) didn't help factor N={N}")
        elif factor == N:
            print(
                f"Darn, looks like r={r} is not the period of x={x} mod N={N}")
        else:
            print(f"Cool, (x,r)=({x},{r}) did it!")
            break

    assert factor != 1
    assert factor != N
    print(
        f"Found factor {factor} of {N} in {tot_invocations} invocations of the circuit")
    print(f"Total time: {tot_time} s")
    print(f"Average time per invocation: {tot_time/tot_invocations} s")


def main(N, extend=False):
    start_time_inc_setup = time.perf_counter()
    # randint(a,b) samples from [a,b] inclusive
    print(f"Trying to factor {N}")
    run_shor(N, extend)
    end_time_inc_setup = time.perf_counter()
    print(
        f"Total run time including setup: {end_time_inc_setup-start_time_inc_setup:3f} s")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        N = int(sys.argv[1])
        main(N)
    elif len(sys.argv) == 3 and sys.argv[1] == '--extended':
        N = int(sys.argv[2])
        main(N, True)
    else:
        print(
            "Usage: python3 shor.py [--extended] N\nWhere N is the number to factor")
        exit(1)
