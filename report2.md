WE HAVE GRAPHS! THEY ARE INCLUDED IN SEPRATE FILES

# Shor's algorithm
Our implementation is **not based on the Google tutorial** but rather on Scott Aaronson's notes (see citations). 
We consider this an extension of the original question since this allows us to use significantly fewer qubits.
With 14 qubits, we are able to factor numbers up to 63.
Additionally, we have extended Shor to be able to use periods that are not even but instead divisible by 3.

## Design & Evaluation 
My implementation was parameterized in input size as follows: For input N, let n = floor(lg N) be the number of bits
in the binary representation of n. Then we use at most 3n qubits (while Google requires 3n+3). The design allows for fewer qubits to be used at the expense of decreasing success probability of the period finding. Specifically, I think of the 3n qubits as being divided into two registers. The first is ideally of size 2n but can be smaller, as long as it is strictly larger than n. The second must be exactly of size n. Here is a table of the sizes I used for interesting values of N, since my computer can only handle 14 qubits with cirq's built-in simulator.

| N (range) | n   | r1  | r2  | Total # qubits |
| --------- | --- | --- | --- | -------------- |
| 8-15      | 4   | 8   | 4   | 12             |
| 16-31     | 5   | 9   | 5   | 14             |
| 32-63     | 6   | 8   | 6   | 14             |

Within the classical part of Shor's algorithm, there are a number of simple number-theoretic helper functions needed: gcd, is_prime, is_power. I tested these separately.
The classical part also includes my extension to allow orders that are not
even but instead divisible by 3.

The quantum part of Shor's algorithm is the period finder. For the period finder, following Aaronson, I implemented a 
unitary matrix representation U_f for the function f(r) = x^r mod N, where x is the random element of U_N chosen classically.
Specifically, I think of the qubits as being divided into two registers.
Then U_f(|r>|y>) = |r>|y XOR (x^r mod N)>. We then observe the second register, essentially committing to a value x^r1.
The state of the first register becomes an evenly weighted combination of |r1 + s>, |r1 + 2s>, etc, where s is the period of x. We then apply the QFT to and observe the first register. Since before QFT the superposition of states are spread out essentially with period s, after QFT we should have almost all amplitude on states r0 such that r0s is a multiple of Q, i.e. r0s = kQ. So k/s ~= r0/Q. Note thats s < N/2 if N is composite while Q >> N. So we find the closest fraction to r0/Q with denominator < N/2. Unfortunately we can be wrong when gcd(k,s) != 1, which is why we check to make sure that x^s actually equals 1 mod N. 

Here is the output for some interesting test cases:
```
$ for x in 1 2 3 4 15 21 33 45 63 ; do python3 shor.py $x ; done
Trying to factor 1
Boo, 1 is prime or 1!
Total run time including setup: 0.000460 s

Trying to factor 2
Boo, 2 is even!
Total run time including setup: 0.000793 s

Trying to factor 3
Boo, 3 is prime or 1!
Total run time including setup: 0.000673 s

Trying to factor 4
Boo, 4 is a 2nd/rd/th power
Total run time including setup: 0.001071 s

Trying to factor 15
Running period finder: trying 13
Using 8 qubit input register and 4 ancillary qubits
--SIMULATING--
Got QFT result: 0/256
Guessing denominator 1
--SIMULATING--
Got QFT result: 64/256
Guessing denominator 4
Cool, (x,r)=(13,4) did it!
Found factor 3 of 15 in 2.0 invocations of the circuit
Total time: 0.8748916999902576 s
Average time per invocation: 0.4374458499951288 s
Total run time including setup: 0.883601 s

Trying to factor 21
Running period finder: trying 11
Using 9 qubit input register and 5 ancillary qubits
--SIMULATING--
Got QFT result: 82/512
Guessing denominator 6
Cool, (x,r)=(11,6) did it!
Found factor 7 of 21 in 1.0 invocations of the circuit
Total time: 22.304759299993748 s
Average time per invocation: 22.304759299993748 s
Total run time including setup: 22.315247 s

Trying to factor 33
Running period finder: trying 8
Using 8 qubit input register and 6 ancillary qubits
--SIMULATING--
Got QFT result: 102/256
Guessing denominator 5
--SIMULATING--
Got QFT result: 77/256
Guessing denominator 10
Darn, (x,r)=(8,10) didn't help factor N=33
Running period finder: trying 1
Using 8 qubit input register and 6 ancillary qubits
--SIMULATING--
Got QFT result: 0/256
Guessing denominator 1
Darn, (x,r)=(1,1) didn't help factor N=33
Running period finder: trying 31
Using 8 qubit input register and 6 ancillary qubits
--SIMULATING--
Got QFT result: 154/256
Guessing denominator 5
Darn, (x,r)=(31,5) didn't help factor N=33
Running period finder: trying 5
Using 8 qubit input register and 6 ancillary qubits
--SIMULATING--
Got QFT result: 77/256
Guessing denominator 10
Cool, (x,r)=(5,10) did it!
Found factor 11 of 33 in 5.0 invocations of the circuit
Total time: 70.96561849999125 s
Average time per invocation: 14.19312369999825 s
Total run time including setup: 70.995195 s

Trying to factor 45
Running period finder: trying 13
Using 8 qubit input register and 6 ancillary qubits
--SIMULATING--
Got QFT result: 171/256
Guessing denominator 3
--SIMULATING--
Got QFT result: 171/256
Guessing denominator 3
--SIMULATING--
Got QFT result: 21/256
Guessing denominator 12
Cool, (x,r)=(13,12) did it!
Found factor 9 of 45 in 3.0 invocations of the circuit
Total time: 40.763321600010386 s
Average time per invocation: 13.587773866670128 s
Total run time including setup: 40.773549 s

Trying to factor 63
Running period finder: trying 10
Using 8 qubit input register and 6 ancillary qubits
--SIMULATING--
Got QFT result: 213/256
Guessing denominator 6
Cool, (x,r)=(10,6) did it!
Found factor 9 of 63 in 1.0 invocations of the circuit
Total time: 13.53432099998463 s
Average time per invocation: 13.53432099998463 s
Total run time including setup: 13.540325 s
```

## Extension demo:

We show the extension in action:

```
$ python3 shor.py --extended 57
Trying to factor 57
Running period finder: trying 1
Using 8 qubit input register and 6 ancillary qubits
--SIMULATING--
Got QFT result: 0/256
Guessing denominator 1
Darn, (x,r)=(1,1) didn't help factor N=57
Running period finder: trying 1
Using 8 qubit input register and 6 ancillary qubits
--SIMULATING--
Got QFT result: 0/256
Guessing denominator 1
Darn, (x,r)=(1,1) didn't help factor N=57
Running period finder: trying 55
Using 8 qubit input register and 6 ancillary qubits
--SIMULATING--
Got QFT result: 142/256
Guessing denominator 9
Cool, (x,r)=(55,9) did it!
Found factor 3 of 57 in 3.0 invocations of the circuit
Total time: 42.83884580002632 s
Average time per invocation: 14.27961526667544 s
Total run time including setup: 42.853002 s
```

## Graph
Here is a graph showing the preformence of Shor's: 
![Graph](images/simon_graph.jpg)

The preformence graph looks somwwhat skewed because the first number that the algorithm can operate on is 15, since it is the first odd composite number that's not a prime power (As you know Shor's does not work on prime powers). 

## README 
- Simply run
```
$ python3 shor.py N
```
Where N is the number you wish to factor.
If N will not be able to use a quantum circuit, then you will see a message explaining why not.
Additionally, you should see output about the circuit as above.

- If you want to use periods divisible by 3, you can try adding the --extended option:
```
$ python3 shor.py --extended N
```
Although this only helps if you use N=57 and you get lucky.

## Citations
Scott Aaronson: "Shor's Algorithm and Hidden Subgroup Problem", 2010.
https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-845-quantum-complexity-theory-fall-2010/lecture-notes/MIT6_845F10_lec07.pdf


# QAOA
Our implementation is an instructed generalisation of https://quantumai.google/cirq/tutorials/qaoa. 
## Design & Evaluation 
- the solution was paramterisized in $n$ in the following steps:    
    - I built the module to parse the arbitary adjacency matrix and check it's validity. This was mostly a matter of string parsing, which in python, is a breeze. 
    - then we create the graph from he adjacency matrix, where `networkx` takes care of most of the work
    - computing QAOA on our completed graph is just a matter of using Cirq, for which paramterisizing about $n$ does not take any work
- To test, I added a method where we compute the random cut on a different integer. To visualise, I chose to work with the maxCut problem, which is natively much more visual. Then, using existing python code, I was able to print out the max cut (althoug for too-dense or too-large graph those pictures become intengible)
- There are quite a few paramters that the code scales with. First we have the obvious parameter which is the number of nodes, $n$ and closely related, the number of edges $e$. Third, the densensess of our scan for the optimal alpha & beta, $g$. Fourth is our number of times that we sample from the circuit, $s$.  
    - for our testing, we kept everything constant but $n$ and made $e$ as large as possible, that is, $nCr(n,2)$
```
$ python QAOA.py
you have more then 16 nodes in your graph, we will run the program for you, but note that it might take ridicoulusly long
        FINAL RESULTS FOR CIRCUITS OF SIZE 3 TO SIZE 16
-------------------------------------------------------------------------

For n=3 we have a total circuit runtime 0.009592399990651757 and total runtime 8.79044119999162 which means we spent 0.0010912307781162226 of the time runnign the circuit
        QAOA found a best cut of size 8.15 while random search found a best cut of size 8.15, for relative score of 1.0.
For n=4 we have a total circuit runtime 0.007962900010170415 and total runtime 8.691639900003793 which means we spent 0.0009161562261877578 of the time runnign the circuit
        QAOA found a best cut of size 21.36 while random search found a best cut of size 21.36, for relative score of 1.0.
For n=5 we have a total circuit runtime 0.013655299990205094 and total runtime 8.33949399998528 which means we spent 0.0016374254829164933 of the time runnign the circuit
        QAOA found a best cut of size 45.28 while random search found a best cut of size 45.28, for relative score of 1.0.
For n=6 we have a total circuit runtime 0.02336039999499917 and total runtime 9.938117700017756 which means we spent 0.0023505859660886723 of the time runnign the circuit
        QAOA found a best cut of size 38.02 while random search found a best cut of size 38.02, for relative score of 1.0.
For n=7 we have a total circuit runtime 0.03707679998478852 and total runtime 11.999338699999498 which means we spent 0.00308990361150403 of the time runnign the circuit
        QAOA found a best cut of size 64.09 while random search found a best cut of size 63.71, for relative score of 1.0059645267618897.
For n=8 we have a total circuit runtime 0.038802600000053644 and total runtime 13.03106919999118 which means we spent 0.002977698867571044 of the time runnign the circuit
        QAOA found a best cut of size 81.03 while random search found a best cut of size 95.47999999999999, for relative score of 0.8486594051110181.
For n=9 we have a total circuit runtime 0.03185349999694154 and total runtime 15.487297300016508 which means we spent 0.0020567500823340935 of the time runnign the circuit
        QAOA found a best cut of size 90.22 while random search found a best cut of size 93.87, for relative score of 0.9611164376265047.
For n=10 we have a total circuit runtime 0.042056099977344275 and total runtime 16.823179900005925 which means we spent 0.002499890046193316 of the time runnign the circuit
        QAOA found a best cut of size 136.23 while random search found a best cut of size 139.76999999999998, for relative score of 0.9746726765400301.
For n=11 we have a total circuit runtime 0.05332869998528622 and total runtime 23.36404029998812 which means we spent 0.0022825118986510794 of the time runnign the circuit
        QAOA found a best cut of size 137.77 while random search found a best cut of size 133.75000000000006, for relative score of 1.0300560747663547.
For n=12 we have a total circuit runtime 0.0833091999811586 and total runtime 24.160093600017717 which means we spent 0.0034482151170597084 of the time runnign the circuit
        QAOA found a best cut of size 186.40999999999997 while random search found a best cut of size 186.40999999999994, for relative score of 1.0000000000000002.
For n=13 we have a total circuit runtime 0.07420079997973517 and total runtime 29.852951700013364 which means we spent 0.0024855431625443575 of the time runnign the circuit
        QAOA found a best cut of size 218.63000000000005 while random search found a best cut of size 214.01, for relative score of 1.021587776272137.
For n=14 we have a total circuit runtime 0.26831549999769777 and total runtime 27.505412900005467 which means we spent 0.009755007167976178 of the time runnign the circuit
        QAOA found a best cut of size 235.3 while random search found a best cut of size 243.63000000000005, for relative score of 0.9658088084390263.
For n=15 we have a total circuit runtime 0.16439449999597855 and total runtime 33.500845300004585 which means we spent 0.00490717468540879 of the time runnign the circuit
        QAOA found a best cut of size 285.37 while random search found a best cut of size 271.2600000000001, for relative score of 1.0520165155201648.
For n=16 we have a total circuit runtime 0.12808490000315942 and total runtime 32.69602470000973 which means we spent 0.003917445658252188 of the time runnign the circuit
        QAOA found a best cut of size 301.7 while random search found a best cut of size 318.46, for relative score of 0.947371726433461.

        FINAL RESULTS:
-------------------------------------------------------------------------
On average 0.0031011099107717097 of the time in circuits, and qaoa's relative preformence to random was 0.9862324248193276
```

As can be observed, on average QAOA preformed slightly better (that is because we set p=1, higher is not very simulatable) and as the time went on, the portion of the time spent simulating quantum circuits rather than setting up the problem isnatnce increased very rapidly (eponentially)

Below is a graph of the preforemence detailed in the above printed report 
![qaoa_graph.jpg](images/qaoa_graph.jpg)

## README 
- fill in the adjacency matrix of ur choosing in `adjacency_matrices/99`
        - if there are issues, the script should fail or worn you
        - no delimeters or anything; see example currently in file
- set
    ```
    MIN_SIZE = 99
    MAX_SIZE = 99 
    ```
    in the `if... __main__...` statement 
- turn `GRAPHICAL_DISPLAY= True` at the top of the page if you'd like to see visualizations
- run `python QAOA.py`
- enjoy!

(running with defualt should expect output simillar to this -- only simllar becasse of randomness)
```
$ python QAOA.py
        FINAL RESULTS FOR CIRCUITS [99,100)
-------------------------------------------------------------------------

For n=99 we have a total circuit runtime 0.07221340000978671 and total runtime 27.023552099999506 which means we spent 0.0026722393763249216 of the time runnign the circuit
        QAOA found a best cut of size 97.26000000000002 while random search found a best cut of size 88.67000000000002, for relative score of 1.0968760572910794.

        FINAL RESULTS:
-------------------------------------------------------------------------
On average 0.0026722393763249216 of the time in circuits, and qaoa's relative preformence to random was 1.0968760572910794
```
## Citations 
- tutorial I used: https://quantumai.google/cirq/tutorials/qaoa 
- to generate diagonal matrix: https://catonmat.net/tools/generate-symmetric-matrices
