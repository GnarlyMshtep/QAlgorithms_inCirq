"""
This is an implementation of the quantum approximation algorithm as proposed in [the original paper](https://arxiv.org/abs/1411.4028) by Farhi, Goldstone, and Gutmann.
Our implementation draws much guidance from [Google's tutorial](https://quantumai.google/cirq/tutorials/qaoa) and improves it by allowing for a much wider range of graphs to be operated on. This is only possible since our implementation is simulated on a classical computer (where the relative "position" and "connections" between the simulated qubits are not taken into account).
"""

import networkx as nx           # to construct and work with graphs graphs
import matplotlib.pyplot as plt  # F_REMOVED: to draw examples along the way
import numpy as np              # for general numberical manipulations
# for working with alpha and beta, the paramters of the algorithm
import sympy
import cirq                     # for quantum simulation

import cirq_google              # F_REMOVED to set working device below
working_device = cirq_google.Bristlecone


# print("our working device is", working_device)


def generate_problem_instance():
    """
    Whip-up the problem instance. I'll take this as a given, and once everyhting else is working fine and dandy, I'll come back and change this.
    """
    # Set the seed to determine the problem instance.
    np.random.seed(seed=11)

    # Identify working qubits from the device.
    device_qubits = working_device.qubits
    working_qubits = sorted(device_qubits)[:12]

    # Populate a networkx graph with working_qubits as nodes.
    working_graph = nx.Graph()
    for qubit in working_qubits:
        working_graph.add_node(qubit)

    # Pair up all neighbors with random weights in working_graph.
    for qubit in working_qubits:
        for neighbor in working_device.neighbors_of(qubit):
            if neighbor in working_graph:
                # Generate a randomly weighted edge between them. Here the weighting
                # is a random 2 decimal floating point between 0 and 5.
                working_graph.add_edge(
                    qubit, neighbor, weight=np.random.randint(0, 500) / 100
                )

    nx.draw_circular(working_graph, node_size=1000, with_labels=True)
    plt.show()
    return working_graph


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


def do_flag1(qaoa_circuit, working_graph):
    alpha_value = np.pi / 4
    beta_value = np.pi / 2
    sim = cirq.Simulator()

    sample_results = sim.sample(
        qaoa_circuit,
        params={alpha: alpha_value, beta: beta_value},
        repetitions=20_000
    )
    print(f'Alpha = {round(alpha_value, 3)} Beta = {round(beta_value, 3)}')
    print(f'Estimated cost: {estimate_cost(working_graph, sample_results)}')


def search_over_parmspace():
    # Set the grid size = number of points in the interval [0, 2π).
    grid_size = 5

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
    plt.title('Heatmap of QAOA Cost Function Value')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.imshow(exp_values)


def output_cut(S_partition):
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

    nx.draw_circular(
        working_graph,
        node_color=coloring,
        node_size=1000,
        with_labels=True,
        width=weights)
    plt.show()
    size = nx.cut_size(working_graph, S_partition, weight='weight')
    print(f'Cut size: {size}')


def do_qaoa():  # Number of candidate cuts to sample.

    num_cuts = 100
    candidate_cuts = sim.sample(
        qaoa_circuit,
        params={alpha: best_parameters[0], beta: best_parameters[1]},
        repetitions=num_cuts
    )

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

        # If you found a better cut update best_qaoa_cut variables.
        if cut_size > best_qaoa_cut_size:
            best_qaoa_cut_size = cut_size
            best_qaoa_S_partition = S_partition
            best_qaoa_T_partition = T_partition


if __name__ == "__main__":
    alpha = sympy.Symbol('alpha')
    beta = sympy.Symbol('beta')

    working_graph = generate_problem_instance()
    circuit = construct_circuit(working_graph)

    # get best params
    best_exp_index = np.unravel_index(np.argmax(exp_values), exp_values.shape)
    best_parameters = par_values[best_exp_index]
    print(f'Best control parameters: {best_parameters}')
