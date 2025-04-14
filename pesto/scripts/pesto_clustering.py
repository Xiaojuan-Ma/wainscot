"""
This file implements Pesto's clustering algorithm, closely following the edge-merging
criteria in Section 3.3 and Theorem 3.5 of Pesto (Towards optimal placement and scheduling of DNN operations with Pesto).
"""
import os
import copy
import pickle
import random
from collections import deque, defaultdict

import numpy as np
import networkx as nx
from collections import defaultdict


pesto_dir = '../'


# Generate a random DAG with NetworkX: for test purpose
def generate_dag(num_nodes, edge_prob):
    """
    Generates a random directed acyclic graph (DAG) with a specified number of nodes and edge probability.

    Parameters:
    - num_nodes (int): Number of nodes in the graph.
    - edge_prob (float): Probability of an edge existing between any two nodes.

    Returns:
    - dag (networkx.DiGraph): A directed acyclic graph.
    """
    # Start with an empty directed graph
    dag = nx.DiGraph()

    # Add nodes to the graph
    dag.add_nodes_from(range(num_nodes))

    # Randomly add edges with edge_prob probability, ensuring no cycles
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Ensure j > i to avoid cycles
            if np.random.rand() < edge_prob:
                # Add edge with a random 'weight' attribute
                dag.add_edge(i, j, num_bytes=np.random.rand())  # Random weight between 0 and 1

    return dag


# Calculate height for each node in a DAG based on Definition 3.4 in Pesto paper
def calculate_heights(G):
    """
    :param G: a NetworkX DAG
    :return: a heights dictionary. key: node vertex; value: height of this node
    """
    # Extract vertices and edges from the NetworkX DAG
    vertices = list(G.nodes)
    edges = list(G.edges)

    # Initialize in-degree count and adjacency list
    in_degree = {v: 0 for v in vertices}
    adj_list = defaultdict(list)
    heights = {v: 0 for v in vertices}  # Stores the height of each vertex

    for u, v in edges:
        adj_list[u].append(v)
        in_degree[v] += 1

    # Initialize queue with all vertices having in-degree of 0 (root nodes)
    queue = deque([v for v in vertices if in_degree[v] == 0])

    # Set initial height for root nodes
    for v in queue:
        heights[v] = 1

    # Process vertices level by level
    while queue:
        next_level = deque()  # Prepare to collect nodes for the next level
        for u in queue:
            for v in adj_list[u]:
                in_degree[v] -= 1  # Remove edge u -> v
                if in_degree[v] == 0:
                    next_level.append(v)  # Add vertex v to next level
                    heights[v] = heights[u] + 1  # Set height based on parent height

        # Move to the next level in the DAG
        queue = next_level

    return heights


# One round of merging feasible edges while keeping the graph as a DAG. Based on Theorem 3.5 in Pesto paper
def merge_feasible_edges(G, heights, n_threshold):
    """
    :param G: (nx.DiGraph): The input DAG.
    :param heights: (dict): A dictionary where heights[v] gives the height of node v.
    :param n_threshold: the preferred size of the merged graph

    :return: merged_G (nx.DiGraph): A new DAG with merged nodes.
    """

    def identifying_feasible_edges(G):
        edges_to_merge = []

        # Calculate d_i values for all edges
        d_values = {(u, v): heights[v] - heights[u] for u, v in G.edges}

        # Identify feasible edges for merging
        print('identifying feasible edges......')

        # 'num_bytes' needs to be an edge attribute
        sorting_metric = 'num_bytes'
        # Randomly select an edge
        u, v = random.choice(list(G.edges))
        if sorting_metric in G[u][v]:
            edges = sorted(G.edges(data=True), key=lambda x: x[2][sorting_metric], reverse=True)
        else:
            edges = G.edges(data=True)

        # Iterate over the sorted edges
        added_vertices = set()
        one_dis_separate = True
        recheck_index = 0
        for u, v, data in edges:
            # Check Condition (i): Ensure no repeated vertices in edge pairs (Pesto Theorem 3.5)
            if u in added_vertices or v in added_vertices:
                continue
            # Check Condition (ii)
            d_i = d_values[(u, v)]
            successors_u = list(G.successors(u))
            predecessors_v = list(G.predecessors(v))

            # Check if one of the following subconditions holds:
            condition_2 = False
            if len(successors_u) == 1 or len(predecessors_v) == 1:
                edges_to_merge.append((u, v))
            elif one_dis_separate:
                if heights[v] == heights[u] + 1:
                    edges_to_merge.append((u, v))
            else:
                valid = True
                for w in successors_u:
                    if w != v and heights[w] <= heights[u] + d_i:
                        valid = False
                        break
                if valid:
                    edges_to_merge.append((u, v))
                    one_dis_separate = False

            # Check Condition (iii): Ensure no height conflicts between pairs
            discard = False
            condition_3 = True
            for (u_j, v_j) in edges_to_merge:
                d_j = d_values[(u_j, v_j)]
                if heights[u] == heights[v_j] + d_j and (u, v_j) in G.edges:
                    edges_to_merge.pop()
                    discard = True
                    # print(f'failed at condition 3')
                    condition_3 = False
                    if one_dis_separate:
                        recheck_index -= 1
                    break
            if not discard:
                added_vertices.add(u)
                added_vertices.add(v)


        if recheck_index > 0:
            indices_to_remove = []
            for i in range(recheck_index):
                u, v = edges_to_merge[i]
                valid = True
                successors_u = list(G.successors(u))
                d_i = d_values[(u, v)]
                for w in successors_u:
                    if w != v and heights[w] <= heights[u] + d_i:
                        valid = False
                        break
                if not valid:
                    indices_to_remove.insert(0, i)
            for index in indices_to_remove:  # already in decreasing order
                del edges_to_merge[index]

        # print(f'number of feasible edges {len(edges_to_merge)}')
        return edges_to_merge


    def add_edge_with_sum(_G, u, v, target_attributes=[], attributes={}):
        # Check if the edge already exists
        if _G.has_edge(u, v):
            for attribute in target_attributes:
                # If 'num_bytes' exists in both the edge and attributes, sum them up
                if attribute in _G[u][v] and attribute in attributes:
                    _G[u][v][attribute] += attributes[attribute]
                # If 'num_bytes' exists only in attributes, add it to the edge
                elif attribute in attributes:
                    _G[u][v][attribute] = attributes[attribute]
                # Update any other attributes as needed (without overwriting existing ones unless specified)
            for key, value in attributes.items():
                if key not in target_attributes:
                    _G[u][v][key] = value
        else:
            # If the edge doesn't exist, add it with the given attributes
            _G.add_edge(u, v, **attributes)
        return _G

    def merge_edges(G, edges_to_merge):
        """
        Updates the input G in place. if want a copy, make change
        :param G: the digraph (DAG) that want to merge
        :param edges_to_merge: collections of edges that will be merged
        :return: updated G with merged nodes and updated edge info
        """
        merged_nodes = {}
        for u, v in edges_to_merge:
            # Create a new "super node"
            super_node = f"{u}_{v}"
            merged_nodes[u] = super_node
            merged_nodes[v] = super_node

        _G = nx.DiGraph()
        for node in G.nodes:
            if node not in merged_nodes:
                _G.add_node(node)
        _G.add_nodes_from(set(merged_nodes.values()))

        for u, v, edge_attributes in G.edges(data=True):
            # if both end points are not in _G and they have the same map: this edges has been merged
            if u not in _G and v not in _G and merged_nodes[u] == merged_nodes[v]:
                # print(f'edges that skip {u} {v}')
                continue
            start = u if u in _G.nodes else merged_nodes[u]
            end = v if v in _G.nodes else merged_nodes[v]

            # Add the edge with attributes
            # add new edge to _G, if already exist, the num_bytes info will be aggregated
            # notice:only the num_bytes is updated and meaningful!!! if want to use other info, need update accordingly!
            _G = add_edge_with_sum(_G, start, end, target_attributes=['num_bytes'], attributes=edge_attributes)

        G = _G
        return G


    terminate = False

    if len(G.nodes) <= n_threshold:
        print(f'the size of the graph {len(G.nodes)} already achieved the preferred value {n_threshold}')
        terminate = True
        return G, terminate


    edges_to_merge = identifying_feasible_edges(G)
    # print(edges_to_merge)
    if len(edges_to_merge) == 0:
        print('no feasible edges for merging')
        terminate = True
        return G, terminate
    # Merge nodes in edges_to_merge
    G = merge_edges(G, edges_to_merge)

    return G, terminate


def merge_edges_in_batches(G, feasible_edges):
    # Dictionary to track super nodes: maps original nodes to their super node
    node_map = {}

    for u, v in feasible_edges:
        # Update u and v to the super nodes if they have been merged previously
        u = node_map.get(u, u)
        v = node_map.get(v, v)

        # Skip if u and v are already the same super node or if the edge does not exist
        if u == v or not G.has_edge(u, v):
            continue

        # Create the super node label
        super_node_label = f"{u}_{v}"
        # print(f"Creating super node {super_node_label}")

        # Add the new super node
        G.add_node(super_node_label)

        # Redirect predecessors of u and successors of v to the super node
        for pred in G.predecessors(u):
            if pred != v:
                G.add_edge(pred, super_node_label)
        for succ in G.successors(v):
            if succ != u:
                G.add_edge(super_node_label, succ)

        # Remove u and v from the graph
        G.remove_node(u)
        G.remove_node(v)

        # Update the node map to reflect the merge
        node_map[u] = super_node_label
        node_map[v] = super_node_label


def iterative_clustering(G, n_threshold):
    """
    iteratively coarsening the input G until the specified the graph size is achieved or no feasible edges exist
    :param G: networx digraph, DAG
    :param n_threshold: the preferred resultant graph size
    :return: a new graph merged_G with updated nodes and edges. input G is untouched
            1. if an edge (u, v) is merged, the new nodes will have name 'u_v'
                notice the merging will be applied iteratively, therefore it is possible to have 'u_v_s...'
            2. no multiple edges. if after merging, there are multiple edges between nodes, num_bytes will aggregated
    """
    # Check if G is a DAG
    is_dag = nx.is_directed_acyclic_graph(G)
    assert is_dag, 'the input computation graph is not DAG'
    print("before merge: The graph is a DAG:", is_dag)
    print("before merge: graph nodes :", len(G.nodes()))
    print("before merge: graph edges :", len(G.edges()))
    print('merging......')
    merged_G = copy.deepcopy(G)   # use deep copy to avoid accident change of the input G
    cnt = 0
    while True:
        cnt += 1
        print(f'--------------merging round {cnt}---------')
        heights = calculate_heights(merged_G)
        # merged_G will be updated in-place
        merged_G, terminate = merge_feasible_edges(merged_G, heights, n_threshold)
        if terminate:
            break
        print("after merge: graph nodes:", len(merged_G.nodes()))
        print("after merge: graph edges ", len(merged_G.edges()))
        # Check if G is a DAG
        is_dag = nx.is_directed_acyclic_graph(merged_G)
        if not is_dag:  # for debug
            # find one circle
            # try:
            #     cycle = nx.find_cycle(merged_G)
            #     print("Cycle found:", cycle)
            # except nx.exception.NetworkXNoCycle:
            #     print("No cycles found")

            # find all circles:
            cycles = list(nx.simple_cycles(merged_G))
            for idx, cycle in enumerate(cycles, start=1):
                print(f"Cycle {idx}: {cycle}")

        assert is_dag, 'the graph after merge is not DAG'

    return merged_G

def load_baechi_graph(op_graph_file):
    with open(op_graph_file, 'rb') as file:
        op_graph = pickle.load(file)
    is_dag = nx.is_directed_acyclic_graph(op_graph)
    print(f'is op_graph a dag {is_dag}')
    print(f'num of nodes {len(op_graph.nodes)} sample of nodes {op_graph.nodes[0]}')
    edges = list(op_graph.edges)
    print(f'num of edges {len(op_graph.edges)} sample of edges {op_graph.edges[edges[0]]}')
    # Set default value for 'num_bytes' attribute where missing
    for u, v, data in op_graph.edges(data=True):
        if 'num_bytes' not in data:
            data['num_bytes'] = 0  # Assign your desired default value

    return op_graph


def run_pesto_clustering(model_name, pesto_dir, surfix=''):
    # the .pkl file needs to be in the parent folder of current script
    merged_G_pkl = pesto_dir + f'{model_name}_merged_G_forward{surfix}.pkl'


    # if os.path.exists(merged_G_pkl):
    #     print(f"{merged_G_pkl} exists, exit without clustering runs.")
    #     exit()
    # else:
    #     print(f" do clustering.")


    op_graph_file = pesto_dir + f'{model_name}_colocation_graph_for_pesto_forwardTrue_colocatebackTrue.pkl'

    G = load_baechi_graph(op_graph_file)

    n_threshold = 200

    # Apply the merging algorithm
    merged_G = iterative_clustering(G, n_threshold)

    write_file = True
    if write_file:
        with open(merged_G_pkl, 'wb') as file:
            pickle.dump(merged_G, file)
            print(f'the merged graph has been written to {merged_G_pkl}')



if __name__ == '__main__':
    # model names currently support, change name format if want to include more info
    model_name = 'gnmt_v2'  # 'pnasnet_mobile'， 'nasnet_mobile'， 'transformer', 'gnmt_v2'
    run_pesto_clustering(model_name, pesto_dir)
