"""
This is an implementation of the quantum approximation algorithm as proposed in [the original paper](https://arxiv.org/abs/1411.4028) by Farhi, Goldstone, and Gutmann.
Our implementation draws much guidance from [Google's tutorial](https://quantumai.google/cirq/tutorials/qaoa) and improves it by allowing for a much wider range of graphs to be operated on. This is only possible since our implementation is simulated on a classical computer (where the relative "position" and "connections" between the simulated qubits are not taken into account).
"""

from cmath import sin
import random
import time                   # to sample a random partition to check against QAOA
import networkx as nx           # to construct and work with graphs graphs
import matplotlib.pyplot as plt  # F_REMOVED: to draw examples along the way
import numpy as np              # for general numberical manipulations
# for working with alpha and beta, the paramters of the algorithm
import sympy
import cirq                     # for quantum simulation

import cirq_google              # F_REMOVED to set working device below
working_device = cirq_google.Bristlecone


GRAPHICAL_DISPLAY = True


def is_not_proper_adjacency_matrix(l):
    if not (isinstance(l, list) and isinstance(l[0], list)):
        print(
            'The input read is 1 dimensional or has no dimension at all. We need a matrix.')
        return True

    first_row_len = len(l[0])
    if(first_row_len >= 16):
        print('you have more then 16 nodes in your graph, we will run the program for you, but note that it might take ridicoulusly long')
    # check that it's a matrix and that it's square
    for row in l:
        if len(row) != first_row_len:
            print("the row", row, 'length is not', first_row_len,
                  'as it should be. This is not a matrix, try again!')
            return True
    for i in range(first_row_len):
        for j in range(first_row_len):
            if i == j and l[i][j] != 0:
                print('The node', i, 'has an edge of weight',
                      l[i][j], 'with itself. In the context of this problem, it doesn\'t make sense.')
                return True
            if l[i][j] != l[j][i]:
                print('The matrix at position', (i, j), 'does not equal the matrix in position',
                      (j, i), '. Remember, this graph must be undirected.')
                return True
    return False


def msinumeric(x):
    if len(x) == 0:
        return False
    for char in x:
        if char not in '0,1,2,3,4,5,6,7,8,9,.,-'.split(','):
            return False
    return True


def rem_newline_and_return_self(lst):
    y = [x for x in lst if msinumeric(x)]
    # print(y)
    return y


def generate_problem_instance(n):
    """
    Whip-up the problem instance. I'll take this as a given, and once everyhting else is working fine and dandy, I'll come back and change this.
    """
    with open(f'adjacency_matrices/{n}', 'r') as f:
        lines = f.readlines()
        l = [[float(num) for num in rem_newline_and_return_self(line.strip().split(' '))]
             for line in lines]
    # print(l)
    # for now, it is the user's responsibility to create a proper adjacency matrix
    if is_not_proper_adjacency_matrix(l):
        print('As the last print indicated, your adjacency matrix is not proper.\n Exiting the program...')
        exit(1)

  # Populate a networkx graph with working_qubits as nodes.
    working_graph = nx.Graph()
    for i in range(len(l[0])):  # add as many nodes as the adjacency matrix requests
        working_graph.add_node(cirq.LineQubit(i))

    # Pair up all neighbors with random weights in working_graph.
    for i in range(len(l[0])):
        # we must only iterate up to the diagonal since the matrix is symetric about it
        for j in range(i, len(l[0])):
            if l[i][j] > 0:
                working_graph.add_edge(
                    cirq.LineQubit(i), cirq.LineQubit(j), weight=l[i][j]
                )
    num_nodes = working_graph.number_of_nodes()
    if GRAPHICAL_DISPLAY:
        plt.title(f'Your original graph, for size {num_nodes}:')
        nx.draw_spring(working_graph, node_size=1000, with_labels=True)
        # nx.draw_networkx_edge_labels(
        #   working_graph, pos=nx.spring_layout(working_graph))
        # print(working_graph.edges(data=True))
        plt.show()
    return working_graph, num_nodes


def construct_circuit(working_graph, alpha, beta):
    """ construct circuit from working graph """

    # Symbols for the rotation angles in the QAOA circuit.

    qaoa_circuit = cirq.Circuit(
        # Prepare uniform superposition on working_qubits == working_graph.nodes
        cirq.H.on_each(working_graph.nodes()),

        # Do ZZ operations between neighbors u, v in the graph. Here, u is a qubit,
        # v is its neighboring qubit, and w is the weight between these qubits.
        (cirq.ZZ(u, v) ** (alpha * w['weight'])
         for (u, v, w) in working_graph.edges(data=True)),

        # Apply X operations along all nodes of the graph. Again working_graph's
        # nodes are the working_qubits. Note here we use a moment
        # which will force all of the gates into the same line.
        cirq.Moment(cirq.X(qubit) ** beta for qubit in working_graph.nodes()),

        # All relevant things can be computed in the computational basis.
        (cirq.measure(qubit) for qubit in working_graph.nodes()),
    )
    return qaoa_circuit


def estimate_cost(graph, samples):
    """
    Estimate the cost function of the QAOA on the given graph using the
    provided computational basis bitstrings.


    I STILL HAVE NO IDEA WHAT THIS DOES
    *this may be something which I shouldn't do? I think it for graphs with no definite edge weights and whatnot?*
    """
    cost_value = 0.0

    # Loop over edge pairs and compute contribution.
    for u, v, w in graph.edges(data=True):
        u_samples = samples[str(u)]
        v_samples = samples[str(v)]

        # Determine if it was a +1 or -1 eigenvalue.
        u_signs = (-1)**u_samples
        v_signs = (-1)**v_samples
        term_signs = u_signs * v_signs

        # Add scaled term to total cost.
        term_val = np.mean(term_signs) * w['weight']
        cost_value += term_val

    return -cost_value


# def do_flag1(qaoa_circuit, working_graph):
#    """
#    This seems to be just try running the circuit on some arbitary alpha beta
#    """
#
#    alpha_value = np.pi / 4
#    beta_value = np.pi / 2
#    sim = cirq.Simulator()
#
#    sample_results = sim.sample(
#        qaoa_circuit,
#        params={alpha: alpha_value, beta: beta_value},
#        repetitions=20_000
#    )
#    print(f'Alpha = {round(alpha_value, 3)} Beta = {round(beta_value, 3)}')
#    print(f'Estimated cost: {estimate_cost(working_graph, sample_results)}')


def search_over_parmspace_for_best_alpha_beta(sim, qaoa_circuit, grid_size):
    # Set the grid size = number of points in the interval [0, 2π).

    exp_values = np.empty((grid_size, grid_size))
    par_values = np.empty((grid_size, grid_size, 2))

    for i, alpha_value in enumerate(np.linspace(0, 2 * np.pi, grid_size)):
        for j, beta_value in enumerate(np.linspace(0, 2 * np.pi, grid_size)):
            samples = sim.sample(
                qaoa_circuit,
                params={alpha: alpha_value, beta: beta_value},
                repetitions=20000
            )
            exp_values[i][j] = estimate_cost(working_graph, samples)
            par_values[i][j] = alpha_value, beta_value
    if GRAPHICAL_DISPLAY:
        plt.title('Heatmap of QAOA Cost Function Value')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        plt.imshow(exp_values)
        plt.show()
    return exp_values, par_values


def visualise_cut(S_partition, working_graph, plt_title_string):
    """Plot and output the graph cut information.

    could be eventually removed
    """

    # Generate the colors.
    coloring = []
    for node in working_graph:
        if node in S_partition:
            coloring.append('blue')
        else:
            coloring.append('red')

    # Get the weights
    edges = working_graph.edges(data=True)
    weights = [w['weight'] for (u, v, w) in edges]
    if GRAPHICAL_DISPLAY:
        plt.title(plt_title_string)
        nx.draw_circular(
            working_graph,
            node_color=coloring,
            node_size=1000,
            with_labels=True,
            width=weights)
        plt.show()
    size = nx.cut_size(working_graph, S_partition, weight='weight')
    #print(f'Cut size: {size}')
    return size


# Number of candidate cuts to sample.
def do_qaoa(sim, working_graph, qaoa_circuit, alpha_best, beta_best, num_cuts):
    start_circ_time = time.perf_counter()
    candidate_cuts = sim.sample(
        qaoa_circuit,
        params={alpha: alpha_best, beta: beta_best},
        repetitions=num_cuts
    )
    circ_runtime = time.perf_counter() - start_circ_time
    # Variables to store best cut partitions and cut size.
    best_qaoa_S_partition = set()
    best_qaoa_T_partition = set()
    best_qaoa_cut_size = -np.inf

    # Analyze each candidate cut.
    for i in range(num_cuts):
        candidate = candidate_cuts.iloc[i]
        one_qubits = set(candidate[candidate == 1].index)
        S_partition = set()
        T_partition = set()
        for node in working_graph:
            if str(node) in one_qubits:
                # If a one was measured add node to S partition.
                S_partition.add(node)
            else:
                # Otherwise a zero was measured so add to T partition.
                T_partition.add(node)

        cut_size = nx.cut_size(
            working_graph, S_partition, T_partition, weight='weight')
       # print(i, 'the cut size:', cut_size)
        # If you found a better cut update best_qaoa_cut variables.
        if cut_size > best_qaoa_cut_size:
            best_qaoa_cut_size = cut_size
            best_qaoa_S_partition = S_partition
            best_qaoa_T_partition = T_partition

    return best_qaoa_S_partition, best_qaoa_cut_size, circ_runtime


def get_best_random_cut_out_of(num_cuts):
    best_random_S_partition = set()
    best_random_T_partition = set()
    best_random_cut_size = -9999

    # Randomly build candidate sets.
    for i in range(num_cuts):
        S_partition = set()
        T_partition = set()
        for node in working_graph:
            if random.random() > 0.5:
                # If we flip heads add to S.
                S_partition.add(node)
            else:
                # Otherwise add to T.
                T_partition.add(node)

        cut_size = nx.cut_size(
            working_graph, S_partition, T_partition, weight='weight')

        # If you found a better cut update best_random_cut variables.
        if cut_size > best_random_cut_size:
            best_random_cut_size = cut_size
            best_random_S_partition = S_partition
            best_random_T_partition = T_partition
    return best_random_S_partition


if __name__ == "__main__":
    qaoa_cut_sizes = []
    random_cut_sizes = []
    circ_runtimes = []
    tot_runtimes = []
    MIN_SIZE = 99
    MAX_SIZE = 99
    for n in range(MIN_SIZE, MAX_SIZE+1):
        start_tot_time = time.perf_counter()

        alpha = sympy.Symbol('alpha')
        beta = sympy.Symbol('beta')
        sim = cirq.Simulator()

        working_graph, num_nodes = generate_problem_instance(n)
        qaoa_circuit = construct_circuit(working_graph, alpha, beta)

        # do_flag1(qaoa_circuit, working_graph)

        # get best params
        grid_size = 6  # can increase to get better params
        exp_values, par_values = search_over_parmspace_for_best_alpha_beta(
            sim, qaoa_circuit, grid_size)
        best_exp_index = np.unravel_index(
            np.argmax(exp_values), exp_values.shape)
        best_parameters = par_values[best_exp_index]
        # print(f'Best control parameters for, {n}: {best_parameters}')

        # finally do QAOA
        num_cuts = 20  # increasing this generally makes the random search much better, but the qaoa search only slightly better
        best_qaoa_S_partition, best_qaoa_cut_size, circ_runtime = do_qaoa(
            sim, working_graph, qaoa_circuit, best_parameters[0], best_parameters[1], num_cuts)
        circ_runtimes.append(circ_runtime)

        # print('best of QAOA:', best_qaoa_S_partition, best_qaoa_cut_size)
        # do random guess for comparison
        best_random_S_partition = get_best_random_cut_out_of(num_cuts)

        # print the results

        # print('-----QAOA-----')
        qaoa_cut_sizes.append(visualise_cut(best_qaoa_S_partition, working_graph,
                                            f'The cut as found by QAOA for size {num_nodes}')
                              )
        # print('\n\n-----RANDOM-----')
        random_cut_sizes.append(visualise_cut(best_random_S_partition, working_graph,
                                              f'The cut as found by random searching for size {num_nodes}')
                                )
        tot_runtimes.append(time.perf_counter() - start_tot_time)

    precentege_in_circ = []
    qaoa_rel_to_rand = []
    print(
        f'\tFINAL RESULTS FOR CIRCUITS [{MIN_SIZE},{MAX_SIZE+1})')
    print('-------------------------------------------------------------------------\n')
    for n in range(MAX_SIZE-MIN_SIZE+1):
        precentege_in_circ.append(circ_runtimes[n]/tot_runtimes[n])
        qaoa_rel_to_rand.append(qaoa_cut_sizes[n]/random_cut_sizes[n])
        print(
            f'For n={n+MIN_SIZE} we have a total circuit runtime {circ_runtimes[n]} and total runtime {tot_runtimes[n]} which means we spent {precentege_in_circ[n]} of the time runnign the circuit')
        print(
            f'\tQAOA found a best cut of size {qaoa_cut_sizes[n]} while random search found a best cut of size {random_cut_sizes[n]}, for relative score of {qaoa_rel_to_rand[n]}.')
    print('\n\tFINAL RESULTS:')
    print('-------------------------------------------------------------------------')
    print(f'On average {np.average(np.array(precentege_in_circ))} of the time in circuits, and qaoa\'s relative preformence to random was {np.average(np.array(qaoa_rel_to_rand))}')
