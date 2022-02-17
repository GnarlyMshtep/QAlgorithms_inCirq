import cirq
import numpy as np


# bit ops
def add(x, y):
    """The sum of two bits, mod 2."""
    return (x + y) % 2


def multiply(x, y):
    """Bitwise multiplication of two ints"""
    ret = 0
    while x or y:
        ret += (x & 1) * (y & 1)
        x >>= 1
        y >>= 1
    return bool(ret % 2)


def result_to_int(result, n):
    a = 0
    for i in range(n):
        [measurements_i] = result.measurements[f"q{i}"]
        if len(set(measurements_i)) != 1:
            raise ValueError("Something went wrong!")
        a = (a << 1) + measurements_i[0]
    return a


def result_to_bitlist(result, n):
    """"qbit 0 (MSB) is first"""
    a = []
    for i in range(n):
        [measurements_i] = result.measurements[f"q{i}"]
        if len(set(measurements_i)) != 1:
            raise ValueError("Something went wrong!")
        a.append(measurements_i[0])
    return a


def bitstring_list_to_int(bitstring_list):
    num = 0
    for i in range(len(bitstring_list)):
        num += bitstring_list[len(bitstring_list)-1-i] * 2**i
    return num
# U_f class


def subtruct_bitstrings(subtruct_from: list, subtructor: list):
    assert(len(subtruct_from) == len(subtructor))
    return [(subtruct_from[i] - subtructor[i]) % 2 for i in range(len(subtructor))]


class U_f(cirq.Gate):
    def __init__(self, f, nInp: int, nAnc: int):
        super(U_f, self)
        self.f = f  # this is the function that we will be finding U_f for
        self.nInp = nInp  # these are the number of input bits to the function
        self.nAnc = nAnc  # these are the number of output bits of the function
        # N*N is the size of the U_f matrix
        self.N = (2**(nInp+nAnc))
        print("is this greater than 16:", self.nInp)

    def _num_qubits_(self):
        return self.nInp+self.nAnc

    def _unitary_(self):
        fEvals = [self.f(i) for i in range(2**self.nInp)]
        return np.array(
            [[(fEvals[col >> self.nAnc] ^ col) == row
              for col in range(self.N)]
                for row in range(self.N)]  # iterate over every location in the N*N matrix,
            #                           at column col and row row we evaluate f on teh input bits (col with the ancila bits shifted out),
            # xor it with the col (which is only the ancilla bits (which are lower)) and if we are on the row representing the ancilla bits that should be set
        )

    def _circuit_diagram_info_(self, args):
        return ["U_f"] * (self.nAnc + self.nAnc)
