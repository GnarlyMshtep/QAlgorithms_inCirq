- CS 238 Report: 4 Algorithms in cirq
- James King, Matan Shtepel, Jacob Zhang
- February 16, 2022

--------

0. EXTENSIONS
- B-Z, D-J: 
  - Since those algorithms were kind-of basic, we felt that we had little to add. 
- Simons: 
  - rather than collecting 4m 


1. DESIGN AND EVALUATION
------------------------

* Implementation of U_f

We implemented U_f by creating a class U_f derived from the cirq.Gate class. Using (https://quantumai.google/cirq/custom_gates) for reference, we were able to override the constructor, _num_qubits_, _unitary_, and _circuit_diagram_info_ methods to appropriately define U_f. Overall, the gate takes in a length n and a function from {0, 1}^n to {0, 1}. The _unitary_ function is defined to return a numpy array which represents the matrix mapping |x>|b> to |x>|b^f(x)>.

Our team actually had two functionally identical but different style implementations of U_f. The implementation used in Bernstein-Vazirani and Deutsch-Josza is slightly more straightforward, while the implementation in utils.py used in Simon and Grover is more concise and also supports arbitrary numbers of ancillary bits as needed for Simon's algorithm. The former implementation is probably a bit more readable; one benefit of this implementation is that the loop of length 2**n makes it clear that the resulting matrix will contain exactly 2**n ones.

* Parametrization in n

My circuits are flexible enough to take any n and provide a solution. By designing U_f as a custom gate, I allow it to take any n as a parameter, and have it construct an appropriately sized matrix. Furthermore, the inherent structure of the Deutsch-Jozsa and Bernstein-Vazirani circuits means that increasing n simply means adding more qubits and applying identical transformations to them.

* Code reuse

For the Deutsch-Jozsa and Bernstein-Vazirani algorithm, the fundamental structure of the solution is extremely similar. The circuit consists of something to initialize the state to |000...1> (using an X gate on the final qubit), applying a Hadamard gate to all qubits, applying U_f, then applying Hadamard gate to the first n qubits and measuring. Thus, all code (apart from the testing/verification code) is virtually identical between these two circuits.

* Testing results

As n grows, the number of possible f's increases significantly. For Deutsch-Jozsa, there will only ever be 2 constant functions (all 0 and all 1). However, the number of balanced functions is 2 ** n choose 2 ** (n - 1), which is not far from 2**(2**n). For Simon's algorithm and Grover's algorithm with an unknown number of 1s, the number of functions is similarly double-exponential and therefore prohibitive even for small n. For Bernstein-Vazirani and one-hot Grover, there will be on the order of 2**n possible functions, so it is possible to generate all possible functions for small n. However for practical reasons, we limited the number of cases for each n to be at most 1000.

For Bernstein-Vazirani and Deutsch-Jozsa, the simulated algorithm should always produce the correct answer and so we simply checked that. For Simon, the way we implemented it, the probability of failure is extremely low, though the number of quantum rounds needed is not constant (but rather expected constant), so we similarly checked that the output was correct for all cases. For our implementation of Grover, even in simulation there is a probability of failure. For one-hot Grover, we computed the expected number of cases which would succeed and then compared this the actual number of cases that had the correct output.

One additional test that we did was to print the circuit generated and make sure it stood up to visual inspection. Here is an example for Grover with 5 qubits and a=1:

(0, 0): ───H───────U_f───H───(0)───H───U_f───H───(0)───H───U_f───H───(0)───H───M('q0')───
                   │         │         │         │         │         │
(1, 0): ───H───────U_f───H───(0)───H───U_f───H───(0)───H───U_f───H───(0)───H───M('q1')───
                   │         │         │         │         │         │
(2, 0): ───H───────U_f───H───(0)───H───U_f───H───(0)───H───U_f───H───(0)───H───M('q2')───
                   │         │         │         │         │         │
(3, 0): ───H───────U_f───H───(0)───H───U_f───H───(0)───H───U_f───H───(0)───H───M('q3')───
                   │         │         │         │         │         │
(4, 0): ───H───────U_f───H───(0)───H───U_f───H───(0)───H───U_f───H───(0)───H───M('q4')───
                   │         │         │         │         │         │
(5, 0): ───X───H───U_f───────X─────────U_f───────X─────────U_f───────X───────────────────

Test case generation for Deutsch-Josza, Bernstein-Vazirani and Grover was simply uniformly random over the space of all inputs. For Deutsch-Josza, we chose a random balanced function by randomly permuting an array of 2**(n-1) 0s and 2**(n-1) 1s. Similarly, for arbitrary-input Grover, we chose a random function with a given number of 1s by sampling that many indices without replacement to set to 1.

Test case generation was the source of a subtle bug in our Grover implementation, where we generated random test cases in a way that caused functions to have a 1/(2**n + 1) chance of being
all 0. This made observed success rates lower than expected for small n.

Execution time depended mainly on n and not on the choice of U_f, as discussed in the next section.

* Scalability

The results we obtained show that as n grows, the time to run the circuit for all algorithms grows exponentially, but executes on the order of millseconds to tens of milliseconds. However, the time to setup and complete cases takes much longer as n grows. The results are as follows, where the table gives seconds per simulation for input size n = 1..10:

   n       1       2       3       4       5       6       7       8       9       10
 D-J  0.0016  0.0017  0.0020  0.0024  0.0029  0.0035  0.0045  0.0065  0.0127   0.0354
 B-V  0.0016  0.0019  0.0028  0.0029  0.0031  0.0038  0.0054  0.0080  0.0158   0.0208
Grov  0.0019  0.0026  0.0039  0.0056  0.0084  0.0130  0.0208  0.0992  0.4390   2.5674

Observe that while all grow exponentially, Grover's runtime grows significantly faster due to the
O(2**(0.5n)) iterations required for n bits input size.


2. INSTRUCTIONS/README
----------------------

Each source file can be run on its own as a Python 3 program. In each source file, there is a wrapper function called run(n, f) that takes as arguments the number of bits of input to the function and the function as a mapping from integers to integers. 

The 'if __name__ == "__main__":' section of each source file gives an example of how to use that file's run() function.

3. COMMENTS ON CIRQ
-------------------

Cirq documentation positives:
- There are extensive and fairly well-written tutorials on advanced algorithms, including Shor's algorithm.
- The documentation is embedded in the source code, so it can be accessed while editing in VS Code.
- The website is visually appealing; it would help convince a non-expert with money to invest in quantum computing.

Negatives:
- Documentation is missing or minimal for many classes and functions.
- It is difficult to search the documentation. I would have hoped Google would do a better job.
- There are few examples within the documentation; I had to go to Stack Overflow to figure out how to use the Gate.controlled() function as there was not enough explanation and no example in Google's docs. This compares unfavorably with e.g. the Python or C++ standard library documentation, which both have extensive examples.