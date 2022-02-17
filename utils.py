import cirq
import numpy as np


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

    def _num_qubits_(self):
        return self.nInp+self.nAnc

    def _unitary_(self):
        mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            mat[self.f(i >> self.nAnc) ^ i, i] = 1
        return mat

    def _circuit_diagram_info_(self, args):
        return ["U_f"] * (self.nInp + self.nAnc)
