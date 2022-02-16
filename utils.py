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
        fEvals = [self.f(i) for i in range(2**self.nInp)]
        return np.array(
            [[(fEvals[col >> self.nAnc] ^ col) == row
              for col in range(self.N)]
                for row in range(self.N)]  # iterate over every location in the N*N matrix,
            #                           at column col and row row we evaluate f on teh input bits (col with the ancila bits shifted out),
            # xor it with the col (which is only the ancilla bits (which are lower)) and if we are on the row representing the ancilla bits that should be set
        )

    def _circuit_diagram_info_(self, args):
        return f"U_f({self.f})"
