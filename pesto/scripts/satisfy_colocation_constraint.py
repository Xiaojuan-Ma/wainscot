"""
This file merges clusters formed by Pesto's clustering algo to satisfy the colocation constraints.

"""
import collections
import copy
import csv
import pickle
from collections import defaultdict

import networkx as nx
import pandas as pd


pesto_dir = '../'

# Union-Find helper functions
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x  # Union


def sample():
    # Sample data
    node_sets = [
        {1, 2, 3},  # Node set 0
        {4, 5, 6},  # Node set 1
        {2, 5},  # Node set 2
        {7, 8},  # Node set 3
    ]
    node_to_colocation = {1: 'A', 2: 'B', 3: 'A', 4: 'C', 5: 'B', 6: 'D', 7: 'E', 8: 'F'}

    # Step 1: Map colocation groups to node sets
    colocation_to_sets = defaultdict(set)
    for idx, node_set in enumerate(node_sets):
        for node in node_set:
            colocation = node_to_colocation.get(node)
            if colocation:
                colocation_to_sets[colocation].add(idx)

    # Step 2: Initialize union-find
    uf = UnionFind(len(node_sets))

    # Step 3: Merge sets connected by colocation groups
    for sets in colocation_to_sets.values():
        sets = list(sets)
        for i in range(1, len(sets)):
            uf.union(sets[0], sets[i])

    # Step 4: Group nodes by their root in union-find
    merged_sets = defaultdict(set)
    for idx, node_set in enumerate(node_sets):
        root = uf.find(idx)
        merged_sets[root].update(node_set)

    # Output merged sets
    merged_node_sets = list(merged_sets.values())
    print("Merged Node Sets:", merged_node_sets)

def build_pesto_cluster_colocation_node_dicts(op_to_pesto_cluster, merged_sets):
    merged_colocation_node_to_pesto_clusters = collections.defaultdict(set)
    pesto_clusters_to_merged_colocation_node = {}
    for colo_node, op_sets in merged_sets.items():
        for op in op_sets:
            pesto_cluster = op_to_pesto_cluster[op]
            merged_colocation_node_to_pesto_clusters[colo_node].add(pesto_cluster)
            if pesto_cluster not in pesto_clusters_to_merged_colocation_node:
                pesto_clusters_to_merged_colocation_node[pesto_cluster] = colo_node
            else:
                assert pesto_clusters_to_merged_colocation_node[pesto_cluster] == colo_node, \
                    'all pesto clusters in the same colocation node should have the same colocation group info'
    return merged_colocation_node_to_pesto_clusters, pesto_clusters_to_merged_colocation_node


def merge_pesto_clustered_with_shared_common_colocation_group(node_sets, uf):

    # Step 1: Initialize the primary union-find for node groups
    # uf_primary = UnionFind(len(node_sets))  # Assuming you have a UnionFind class

    # # Sample node sets
    # node_sets = [
    #     {1, 2, 3},  # Node set 0
    #     {4, 5, 6},  # Node set 1
    #     {2, 5},  # Node set 2 (overlapping with sets 0 and 1 by nodes 2 and 5)
    #     {7, 8}  # Node set 3 (no overlaps)
    # ]

    # Step 2: Map each node_set to a set of groups (roots) it contains
    node_set_to_groups = []
    for node_set in node_sets:
        # uf.find(node) returns a representative op id that in the same colocation group with node
        # groups is the set of colocation groups (representative op ids) that this pesto cluster (node_set) belong to
        groups = {uf.find(node) for node in node_set}
        # sets in node_set_to_groups should correspond to the pesto cluster (node_set) with the same index in node_sets
        node_set_to_groups.append(groups)

    # Step 3: Initialize a second union-find structure for group merging:
    # this uf for each colocation group may in multiple clusters
    group_uf = UnionFind(len(node_set_to_groups))

    # Step 4: Merge sets if they have overlapping groups
    for i in range(len(node_set_to_groups)):
        for j in range(i + 1, len(node_set_to_groups)):
            if node_set_to_groups[i].intersection(node_set_to_groups[j]):
                group_uf.union(i, j)

    # Step 5: Collect merged node sets based on merged groups
    merged_colocation_group_to_op_id_sets = defaultdict(set)
    for idx, node_set in enumerate(node_sets):
        root = group_uf.find(idx)
        merged_colocation_group_to_op_id_sets[root].update(node_set)

    # Convert the merged_colocation_group_to_op_id_sets dictionary to a list for final output
    # final_merged_node_sets = list(merged_colocation_group_to_op_id_sets.values())
    # print("Final Merged Node Sets:", final_merged_node_sets)
    print(f'the number of nodes before merge {len(node_sets)}')
    print(f'the number of nodes after merge {len(merged_colocation_group_to_op_id_sets)}')
    return merged_colocation_group_to_op_id_sets   # the key is just a value that does not have real meaning


def parse_colocation_group_info(original_G=None, original_G_pkl=None, merged_G=None, merged_G_pkl=None):
    # original_G, merged_G = None, None
    # for G, pkl_file in zip([original_G, merged_G], [original_G_pkl, merged_G_pkl]):
    if not original_G:
        with open(original_G_pkl, 'rb') as file:
            original_G = pickle.load(file)

    if not merged_G:
        with open(merged_G_pkl, 'rb') as file:
            merged_G = pickle.load(file)

    # Step 1: Map colocation groups to node sets and prepare list of node sets
    colocation_to_sets = defaultdict(set)
    for node, data in original_G.nodes(data=True):
        # todo: check if each node has a colocation group info
        # Notice, the op_graph here is before Baechi's palcement, the colocation_attribute is a list of colocation groups this node belong to
        # example: 'colocation_group': ['loc:@dataset/OneShotIterator']
        # the op_graph after baechi's placment algo, will be only one name
        # our union find algo will do the same thing
        colocation_groups = data['colocation_group']
        for colocation_group in colocation_groups:
            colocation_to_sets[colocation_group].add(node)

    node_sets = []
    op_to_pesto_cluster = {}
    # todo: check: nodes in merged_G do not have attributes?
    for node in merged_G.nodes():
        """
            is op_graph a dag True
            num of nodes 18050 sample of nodes {'weight': 21, 'name': 'dataset/OneShotIterator', 'id': 0, 'temporary_memory': 0, 'persistent_memory': 0, 'output_memory': [40], 'colocation_group': ['loc:@dataset/OneShotIterator'], 'ready_count': 0, 'topo_order': 0}
            num of edges 32853 sample of edges {'id': 0, 'weight': 134, 'tensor': [{'name': 'dataset/OneShotIterator:0', 'weight': 134, 'num_bytes': 40}]}
            Merges all feasible edges in the DAG G according to Theorem 3.5."""
        if node in original_G.nodes:
            node_sets.append({node})   # if it is in the original G: not a merged node
            op_to_pesto_cluster[node] = node
        else:
            merged_nodes = set(int(num) for num in node.split('_'))
            node_sets.append(merged_nodes)
            for e in merged_nodes:
                op_to_pesto_cluster[e] = node

    # Step 2: Initialize union-find: this uf dues to one op may in multiple colocation groups
    # uf = UnionFind(len(node_sets))
    uf = UnionFind(len(original_G.nodes))

    # Step 3: Merge sets connected by colocation groups
    for sets in colocation_to_sets.values():
        sets = list(sets)
        for i in range(1, len(sets)):
            uf.union(sets[0], sets[i])

    # Step 4: Group nodes by their root in union-find
    merged_colocation_group_to_op_id_sets = merge_pesto_clustered_with_shared_common_colocation_group(node_sets, uf)
    # # merged_colocation_group_to_op_id_sets: key: some colocation group name, value: set of op ids in this colocation group
    # merged_colocation_group_to_op_id_sets = defaultdict(set)
    # for idx, node_set in enumerate(node_sets):
    #     root = uf.find(idx)
    #     merged_colocation_group_to_op_id_sets[root].update(node_set)

    return merged_colocation_group_to_op_id_sets, op_to_pesto_cluster



def dfs_topo_order(G=None):

    # A recursive function used by topologicalSort
    def topologicalSortUtil(u, visited, stack):
        # Mark the current node as visited.
        visited.add(u)

        # Recur for all the vertices adjacent to this vertex
        for v in list(G.neighbors(u)):
            if v not in visited:
                topologicalSortUtil(v, visited, stack)

        # Push current vertex to stack which stores result
        stack.append(u)


    assert G, 'no graph is provided for dfs topo order'
    visited = set()
    stack = []

    # Call the recursive helper function to store Topological
    # Sort starting from all vertices one by one
    for u in G.nodes():
        if u not in visited:
            topologicalSortUtil(u, visited, stack)

    # Print contents of the stack
    # print('stack in original order', stack)
    stack = stack[::-1]
    # print(stack)  # return list in reverse order
    return stack


def adjust_topological_order_based_on_tf_colocation_group(G, colo_to_cluster, cluster_to_colo):

    # 1. Perform a DFS-based topological sort, use previous defined function for consistency
    topological_order = dfs_topo_order(G)

    node_to_topo_idx = {node: idx for idx, node in enumerate(topological_order)}

    # Sort nodes within each colocation group
    sorted_colo_groups = {}
    for colo, nodes in colo_to_cluster.items():
        # Sort nodes in each colocation group by their topological order index
        sorted_nodes = sorted(nodes, key=lambda x: node_to_topo_idx[x])
        sorted_colo_groups[colo] = sorted_nodes

    # Reorder nodes by smallest topological index in each colocation group
    final_order = []
    used_nodes = set()

    colocation_order = []

    for node in topological_order:
        colo = cluster_to_colo.get(node)
        if colo not in used_nodes:
            final_order.extend(sorted_colo_groups[colo])
            used_nodes.add(colo)
            colocation_order.append(colo)
    # print("Adjusted order:", final_order)
    # print(f'corresponding colocation group {colocation_order}')
    # return the adjusted ordered list, entry is the node name of the input graph
    return colocation_order



def merge_colocation_groups_prepare_files(original_G_pkl, merged_G_pkl, output_file=''):
    with open(original_G_pkl, 'rb') as file:
        original_G = pickle.load(file)

    with open(merged_G_pkl, 'rb') as file:
        merged_G = pickle.load(file)

    print(f'the number of ops in original G {len(original_G.nodes)}')

    merged_colocation_group_to_op_id_sets, op_to_pesto_cluster = parse_colocation_group_info(original_G=original_G,
                                                                                             merged_G=merged_G)

    colo_to_cluster, cluster_to_colo = (
        build_pesto_cluster_colocation_node_dicts(op_to_pesto_cluster, merged_colocation_group_to_op_id_sets))

    ordered_colocation_groups = adjust_topological_order_based_on_tf_colocation_group(merged_G, colo_to_cluster,
                                                                                      cluster_to_colo)

    op_ids = [merged_colocation_group_to_op_id_sets[e] for e in ordered_colocation_groups]
    op_nums = [len(e) for e in op_ids]

    total_ops = set()
    for op_id_set in op_ids:
        total_ops.update(op_id_set)

    if not output_file:
        output_file = f'{model_name}_pesto_ordered_info.csv'
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['op_nums', 'op_ids', 'colocation_group'])
        # Write data rows
        for op_num, op_set, group in zip(op_nums, op_ids, ordered_colocation_groups):
            writer.writerow([op_num, op_set, group])
    print(f'file has been written to {output_file}')


def run_pesto_colocation_constraint(model_name, pesto_dir, surfix=''):
    original_G_pkl = pesto_dir + f'{model_name}_colocation_graph_for_pesto_forwardTrue_colocatebackTrue.pkl'
    merged_G_pkl = pesto_dir + f'{model_name}_merged_G_forward{surfix}.pkl'
    output_file = pesto_dir + 'info/' + f'{model_name}_pesto_ordered_info{surfix}.csv'
    merge_colocation_groups_prepare_files(original_G_pkl, merged_G_pkl, output_file=output_file)



if __name__ == "__main__":
    # todo: file path might be hard coded, change if needed!
    model_name = 'gnmt_v2' # 'pnasnet_mobile'， 'nasnet_mobile'， 'transformer', 'gnmt_v2'
    run_pesto_colocation_constraint(model_name, pesto_dir)