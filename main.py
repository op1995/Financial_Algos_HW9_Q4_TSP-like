# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# from collections import Set
import itertools

import networkx
from networkx import DiGraph
import numpy as np


def held_karp(dists, source, targets):
    # credit for base code of Help-Karp to CarlEkerot on github
    # https://github.com/CarlEkerot/held-karp
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Parameters:
        dists: distance matrix
    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)
    start = 0;
    if (source != "0"):
        start = ord(source) - 96  # starting node

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(0, n):
        C[(1 << k, k)] = (dists[start][k], start)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(0, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prevs = bits & ~(1 << k)

                options = []  # these are options to go to all the subset, beside k, then ending at k.
                for m in subset:
                    if m == k:
                        continue  # this will be an existing trip, ending at m, then going to m again. Not interesting.
                    options.append((C[(prevs, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(options)

    # We are interested in the bits (nodes) that were passed as targets, but those are string, so they need to be converted
    bits = 0;
    targets_values = [];
    for t in targets:
        t_letter_value = ord(t) - 96
        bits |= 1 << t_letter_value

        targets_values.append(t_letter_value)

        # same code in more detail, easier to understand -
        # numer_value_of_t = (ord(t) - 96)                          #a=1
        # t_value_as_power_of_2_in_binary = 1 << numer_value_of_t
        # bits |= 1 << t_value_as_power_of_2_in_binary

    # OLD/ORIGINAL for all nodes (TSP) -
    # We're interested in all bits but the least significant (the start state)
    # bits = (2**n - 1) - 1

    # Calculate optimal cost
    options_final = []
    for k in targets_values:
        options_final.append((C[(bits, k)][0] + dists[k][start], k))
    opt, parent = min(options_final)

    # Backtrack to find full path
    path = []
    for i in range(len(targets)):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(start)
    my_list = list(reversed(path))
    my_list.append(start)

    return opt, my_list


def held_karp_2taxis(dists, source, targets):
    # credit for base code of Help-Karp to CarlEkerot on github
    # https://github.com/CarlEkerot/held-karp
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Parameters:
        dists: distance matrix
    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)
    start = 0;
    if (source != "0"):
        start = ord(source) - 96  # starting node

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(0, n):
        C[(1 << k, k)] = (dists[start][k], start)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(0, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prevs = bits & ~(1 << k)

                options = []  # these are options to go to all the subset, beside k, then ending at k.
                for m in subset:
                    if m == k:
                        continue  # this will be an existing trip, ending at m, then going to m again. Not interesting.
                    options.append((C[(prevs, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(options)

    # We are interested in the bits (nodes) that were passed as targets, but those are string, so they need to be converted
    bits = 0;
    targets_values = [];
    for t in targets:
        t_letter_value = ord(t) - 96
        bits |= 1 << t_letter_value

        targets_values.append(t_letter_value)

        # same code in more detail, easier to understand -
        # numer_value_of_t = (ord(t) - 96)                          #a=1
        # t_value_as_power_of_2_in_binary = 1 << numer_value_of_t
        # bits |= 1 << t_value_as_power_of_2_in_binary

    # OLD/ORIGINAL for all nodes (TSP) -
    # We're interested in all bits but the least significant (the start state)
    # bits = (2**n - 1) - 1

    # Calculate optimal cost
    options_final = []
    for k in targets_values:
        options_final.append((C[(bits, k)][0] + dists[k][start], k))
    opt, parent = min(options_final)

    best_minimum_price = opt
    two_taxis = 0
    bits_first_taxi_outer = 0
    bits_second_taxi_outer = 0
    first_taxi_parent_outer = 0
    second_taxi_parent_outer = 0
    first_taxi_subset = {}
    second_taxi_subset = {}

    # so now we the path using only 1 taxi.
    # now we should check the options for 2 taxis

    for subset_size in range(1, len(targets_values)):  # check all sizes for taxis - from 1 to all (not including all the targets because we already calculated that)
        # if (subset_size==len(targets_values)):
        #     continue
        for subset in itertools.combinations(targets_values, subset_size):

            # create the complemet of the subset - the riders of the other taxi
            subset_complement = list.copy(targets_values)
            for s in subset:
                subset_complement.remove(s)

            first_taxi_bits = 0
            for s in subset:
                first_taxi_bits |= 1 << s

            second_taxi_bits = 0
            for sc in subset_complement:
                second_taxi_bits |= 1 << sc

            first_taxi_options = []
            for s in subset:
                first_taxi_options.append((C[(first_taxi_bits, s)][0] + dists[s][start], s))
            opt_first_taxi, parent_first_taxi = min(first_taxi_options)

            second_taxi_options = []
            for sc in subset_complement:
                second_taxi_options.append((C[(second_taxi_bits, sc)][0] + dists[sc][start], sc))
            opt_second_taxi, parent_second_taxi = min(second_taxi_options)

            if best_minimum_price > opt_first_taxi + opt_second_taxi:  # found a better option then that found untill now
                best_minimum_price = opt_first_taxi + opt_second_taxi
                two_taxis = 1
                first_taxi_parent_outer = parent_first_taxi
                second_taxi_parent_outer = parent_second_taxi
                first_taxi_subset = list(subset)
                second_taxi_subset = list(subset_complement)
                bits_first_taxi_outer = first_taxi_bits
                bits_second_taxi_outer = second_taxi_bits

    if two_taxis == 1:
        first_taxi_path = []
        for i in range(len(first_taxi_subset)):
            first_taxi_path.append(first_taxi_parent_outer)
            new_bits_first_taxi = bits_first_taxi_outer & ~(1 << first_taxi_parent_outer)
            _, first_taxi_parent_outer = C[(bits_first_taxi_outer, first_taxi_parent_outer)]
            bits_first_taxi_outer = new_bits_first_taxi

        first_taxi_path.append(start)
        my_list_first_taxi = list(reversed(first_taxi_path))
        my_list_first_taxi.append(start)

        second_taxi_path = []
        for i in range(len(second_taxi_subset)):
            second_taxi_path.append(second_taxi_parent_outer)
            new_bits_second_taxi = bits_second_taxi_outer & ~(1 << second_taxi_parent_outer)
            _, second_taxi_parent_outer = C[(bits_second_taxi_outer, second_taxi_parent_outer)]
            bits_second_taxi_outer = new_bits_second_taxi

        second_taxi_path.append(start)
        my_list_second_taxi = list(reversed(second_taxi_path))
        my_list_second_taxi.append(start)

        return 2, best_minimum_price, my_list_first_taxi, my_list_second_taxi

    else:
        # Backtrack to find full path
        path = []
        for i in range(len(targets)):
            path.append(parent)
            new_bits = bits & ~(1 << parent)
            _, parent = C[(bits, parent)]
            bits = new_bits

        # Add implicit start state
        path.append(start)
        my_list = list(reversed(path))
        my_list.append(start)

        return 1, opt, my_list, my_list




def Floyed_Warshall_algo(road_graph: DiGraph):
    n = road_graph.number_of_nodes()
    dist_array = np.full((n, n), 9999)
    for i in range(n):
        dist_array[i][i] = 0
    for u, v, weight in road_graph.edges(data="weight"):
        if weight is not None:
            dist_array[u][v] = weight
            dist_array[v][u] = weight
            pass

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist_array[i][j] = min(dist_array[i][j], dist_array[i][k] + dist_array[k][j])
                dist_array[j][i] = dist_array[i][j]

    return dist_array


def optimal_order(road_graph: DiGraph, source: str, targets):
    weights = Floyed_Warshall_algo(road_graph)
    min_weight_path_weight, min_weight_path = held_karp(weights, source, targets)
    print('min_weight_path_weight - ', min_weight_path_weight, '\n min_weight_path - ', min_weight_path)


def optimal_order_2taxis(road_graph: DiGraph, source: str, targets):
    weights = Floyed_Warshall_algo(road_graph)
    number_of_taxis_to_get, min_weight_path_weight, min_weight_path_first_taxi, min_weight_path_second_taxi = held_karp_2taxis(weights, source, targets)

    if(number_of_taxis_to_get == 2):
        print("Get two taxis.")
        print("The optimal road for taxi 1 is ", min_weight_path_first_taxi)
        print("The optimal road for taxi 2 is ", min_weight_path_second_taxi)

    else:
        print("Get one taxi. The optimal order is ", min_weight_path_first_taxi)
        # print('min_weight_path_weight - ', min_weight_path_weight, '\n min_weight_path - ', min_weight_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # G = DiGraph()
    # G.add_nodes_from([0, 1, 2, 3, 4])
    # G.add_edge(0, 1, weight=1)
    # G.add_edge(0, 2, weight=2)
    # G.add_edge(0, 3, weight=3)
    # G.add_edge(0, 4, weight=4)
    # G.add_edge(1, 2, weight=5)
    # G.add_edge(2, 3, weight=3)
    # G.add_edge(3, 4, weight=1)
    # G.add_edge(4, 1, weight=2)
    # optimal_order(G, "0", ["a", "b"])

    G2 = DiGraph()
    G2.add_nodes_from([0, 1, 2])
    G2.add_edge(0, 1, weight=5)
    G2.add_edge(0, 2, weight=5)
    optimal_order_2taxis(G2, "0", ["a", "b"])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
