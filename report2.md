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
```s
## Citations 
- tutorial I used: https://quantumai.google/cirq/tutorials/qaoa 
- to generate diagonal matrix: https://catonmat.net/tools/generate-symmetric-matrices
