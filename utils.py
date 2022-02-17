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


def bitstring_to_int(bitstring_list: list):
    assert(type(bitstring_list) == list)

    num = 0
    for i in range(len(bitstring_list)):
        num += bitstring_list[len(bitstring_list)-1-i] * 2**i
    return num
# U_f class


def subtruct_bitstrings(subtruct_from: list, subtructor: list):
    assert(len(subtruct_from) == len(subtructor))
    return [(subtruct_from[i] - subtructor[i]) % 2 for i in range(len(subtructor))]


def create_qubits(n):
    return [cirq.GridQubit(i, 0) for i in range(n)]


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
        mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            mat[self.f(i >> self.nAnc) ^ i, i] = 1
        return mat

    def _circuit_diagram_info_(self, args):
        return ["U_f"] * (self.nAnc + self.nAnc)


def rref_f2(matrix_orig: np.array):
    if(len(matrix_orig) <= 0):
        return error("I am mad!")

    matrix = matrix_orig.copy()

    num_rows = len(matrix)
    num_cols = len(matrix[0])

    free_cols = np.ones(num_cols, dtype=int)

    for i in range(num_rows):

        leftmost1_row = None
        one_pos = len(matrix[0])

        for j in range(i, num_rows):
            np_nonz_tmp = np.nonzero(matrix[j])[0]
            if len(np_nonz_tmp) == 0:
                pass
            elif np_nonz_tmp[0] < one_pos:
                leftmost1_row = j
                one_pos = np_nonz_tmp[0]

        if(leftmost1_row == None):
            break

        free_cols[one_pos] = 0

        print(i, leftmost1_row)
        matrix[[i, leftmost1_row]] = matrix[[leftmost1_row, i]]

        for j in range(num_rows):
            if j == i:
                continue

            matrix[j] = (matrix[j] + matrix[i]*matrix[j][one_pos]) % 2

    # DO THE OTHER STUFF
    tmp1 = matrix.transpose().dot(matrix.dot(free_cols)) % 2
    return matrix, tmp1 | free_cols


#print('rref2', rref_f2(np.array([[1, 1, 0], [1, 0, 1], [1, 0, 1]])))
