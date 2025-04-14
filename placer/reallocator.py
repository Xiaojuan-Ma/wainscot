from __future__ import absolute_import, division, print_function

import bisect
import heapq
import math
import pickle
import random

import copy
import functools
import json
import operator
import random
import sys
import time
from collections import deque

import networkx as nx
from future.utils import bytes_to_native_str

import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from placer import adjuster as adjuster_lib
from placer import grouper as grouper_lib
from placer import placer_utils
from placer.m_etf import m_etf
from placer.m_sct import m_sct
from placer.m_topo import m_topo
from placer.virtual_scheduler import VirtualScheduler
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import item as gitem
from utils import logger

from placer.placer_utils import humanize_num_bytes
# from train import device_memories
import matplotlib.pyplot as plt
import numpy as np


def get_freq(degree):
    freq = {}
    for n in degree:
        freq[n] = freq.get(n, 0) + 1
    return freq


def get_digraph_degree_freq(G):
    indegree = sorted(d for n, d in G.in_degree())
    outdegree = sorted(d for n, d in G.out_degree())
    indegree_freq = get_freq(indegree)
    outdegree_freq = get_freq(outdegree)
    print("indegree freq:", indegree_freq)
    print("outdegree freq:", outdegree_freq)
    return indegree_freq, outdegree_freq


def get_independent_subgraph(G, new_nodes):
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in new_nodes)
    if SG.is_multigraph():
        SG.add_edges_from((n, nbr, key, d)
                          for n, nbrs in G.adj.items() if n in new_nodes
                          for nbr, keydict in nbrs.items() if nbr in new_nodes
                          for key, d in keydict.items())
    else:
        SG.add_edges_from((n, nbr, d)
                          for n, nbrs in G.adj.items() if n in new_nodes
                          for nbr, d in nbrs.items() if nbr in new_nodes)
    SG.graph.update(G.graph)
    return SG

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

    stack = stack[::-1]
    return stack


def presum(data_list):
    result = [0] * len(data_list)
    result[0] = data_list[0]
    for i in range(1, len(data_list)):
        result[i] = result[i - 1] + data_list[i]
    return result


class Node:
    def __init__(self, id, memory):
        self.id = id
        self.memory = memory

    def __lt__(self, other):
        return self.memory < other.memory

    def __str__(self):
        return "node id: {self.id} and node memory: {self.memory}"


class ReDevice:
    # _device_graph = None
    _op_graph = None

    class Event:
        # event.type, event.ts, event.size, event.eid
        def __init__(self, type, ts, size, eid):
            self.type = type
            self.size = size
            self.eid = eid
            self.ts = ts

        def __lt__(self, other):
            return self.ts < other.ts

    #
    @classmethod
    def set_op_graph(cls, op_graph):
        ReDevice._op_graph = op_graph

    def __init__(self, device_id, ccs_op_ids_list, used_memory=-1, memory_cap=-1):
        self.device_id = device_id
        self.used_memory = used_memory
        self.memory_cap = memory_cap
        # self.left_memory = self.memory_cap - self.used_memory
        self.ccs_on_device, self.ops_on_device = self._init_ccs_ops_on_device(ccs_op_ids_list)
        self.allocated_memory = 0
        self.events = []
        self.event_id = 0

    def __lt__(self, other):
        return self.left_memory > other.left_memory

    def _init_ccs_ops_on_device(self, ccs_op_ids_list):
        ccs_on_device_dic = {}
        ops_on_device_dic = {}
        for cc_op_ids in ccs_op_ids_list:
            # cc_op_list is a set of op_ids
            # ccs_id will be assigned incrementally each time a new CC instance is created
            new_cc = CC(list(cc_op_ids), self.device_id)
            ccs_id = new_cc.get_ccs_id()
            ccs_on_device_dic[ccs_id] = new_cc
            ops_on_device_dic.update(new_cc.get_ops_in_cc())
        return ccs_on_device_dic, ops_on_device_dic

    def set_memory_simple_sum(self):
        m = 0
        for cc_id, cc in self.ccs_on_device.items():
            m += cc.get_memory_simple_sum()
        return m

    def _get_cur_event_id(self):
        self.event_id += 1
        return str(self.device_id) + str(self.event_id)

    # todo: try to discard 4kb
    def add_event(self, size, start, end):
        # todo: need to check how the persistent memory is allocated
        # if size == 4:
        #     size = 0
        # if size == 0 or end == float('inf'):
        #     self.allocated_memory += size
        #     return

        if size == 0 or size == 4:
            return
        # (type, ts, size, eid)
        eid = self._get_cur_event_id()
        self.events.append(self.Event(type="allocate", ts=start, size=size, eid=eid))
        if end != float('inf'):
            self.events.append(self.Event(type="free", ts=end, size=size, eid=eid))

    def allocate_memories(self):
        self.events.sort(key=lambda x: x.ts)
        memory_allocator = MAllocator(self.events, self.memory_cap)
        self.allocated_memory += memory_allocator.run()
        return self.allocated_memory

    def get_device_id(self):
        return self.device_id

    def get_used_memory(self):
        return self.used_memory

    def get_memory_cap(self):
        return self.memory_cap

    def is_device_allocation_feasible(self):
        return self.left_memory >= 0

    def get_ops_on_device(self):
        return self.ops_on_device

    def get_ccs_on_device(self):
        return self.ccs_on_device

    def get_left_memory(self):
        return self.left_memory

    def add_cc(self, cc):
        # cc is a CC object
        self.ccs_on_device[cc.get_ccs_id()] = cc
        self.memory_simple_sum += cc.get_cc_simple_sum_memory()
        # todo: use which one? overestimation or underestimation?
        # maybe over? then remove? max_op_memory is the overestimation part: assume peak at the same time
        # max_op_memory = cc.ops_in_cc[cc.max_op_id].op_memory
        # self._used_memory += cc.get_cc_persistent_memory() + max_op_memory
        cc.set_cc_belong_to_device_id(self.device_id)
        for _, op_object in cc.ops_in_cc.items():
            self._add_op(op_object)

    def remove_cc(self, cc):
        self.ccs_on_device.pop(cc.ccs_id)
        cc.set_cc_belong_to_device_id(-1)
        for _, op in cc.ops_in_cc.items():
            self._remove_op(op)

    def _add_op(self, op_object):
        self.ops_on_device[op_object.op_id] = op_object

    def _remove_op(self, op):
        self.ops_on_device.pop(op.op_id)

    def _update_left_memory(self):
        self.left_memory = self.memory_cap - self._used_memory

    def update_used_memory(self, used_memory):
        self.used_memory = used_memory
        self.update_left_memory()

    def get_ccs_from_ops(self, op_ids):
        # print("device id and peaks op ids:", self._device_id, op_ids)
        # return is a dict: key: ccs_id, value: set of op_ids in this cc
        involved_ccs = {}
        for op_id in op_ids:
            ccs_id = self.ops_on_device[op_id].get_op_belong_to_ccs_id()
            # print("peak op id and belong to ccs _d:", op_id, ccs_id)
            involved_ccs[ccs_id] = involved_ccs.get(ccs_id, set())
            involved_ccs[ccs_id].add(op_id)
            # print(involved_ccs[ccs_id])
        return involved_ccs

    def set_peak_ops(self, peak_ops):
        # peak_ops: set of op ids
        self.peak_ops = peak_ops

    def set_peak_ccs(self):
        self.peak_ccs = self.get_ccs_from_ops(self.peak_ops)

    def get_peak_ccs(self):
        return self.peak_ccs

    # self._used_memory = device_info._node["peak_memory"]
    # self.peak_ops = device_info._node["peak_ops"]
    def update_after_evaluation(self, used_memory, peak_op_ids):
        self.update_used_memory(used_memory)
        self.peak_ops = peak_op_ids
        self.peak_ccs = self.get_ccs_from_ops(peak_op_ids)

    def _remove_branch(self, branch_dict):
        ccs_id = branch_dict["ccs_id"]
        op_ids = branch_dict["branch_op_ids"]
        for op_id in op_ids:
            self._remove_op(op_id)
            self.ccs_on_device[ccs_id].get_ops_in_cc.pop(op_id)

    # todo: branche cascade is not correct
    def cap_one_device(self):
        def _heuristic1():
            # try to keep large ccs and utilize the device the memory as much as possible
            # return is a list of ccs_to_be_removed and the estimated left memory

            def _get_ordered_op_branches(ccs_to_remove):
                _branch_nodes = []
                _branches_dict = {}
                _bid2node = {}
                for ccs_id, cc in ccs_to_remove.items():
                    if len(cc.ops_in_cc) == 1:
                        continue
                    # print("cc ops:", cc.ops_in_cc.keys())
                    peak_ops = [self.ops_on_device[op_id] for op_id in self.peak_ccs[ccs_id]]
                    # print("in branches: ccs id and involved")
                    new_branches_dict = cc.get_cc_branches(peak_ops, ReDevice._op_graph)
                    # print("branches for this cc", new_branches_dict)
                    _branches_dict.update(new_branches_dict)
                    for _, _branch_dict in new_branches_dict.items():
                        branch_node = Node(_branch_dict["branch_id"], _branch_dict["branch_memory"])
                        _branch_nodes.append(branch_node)
                        _bid2node[_branch_dict["branch_id"]] = branch_node
                _branch_nodes.sort()
                print("sorted branches")
                for branch in _branch_nodes:
                    print("branch:", _branches_dict[branch.id])
                return _branches_dict, _branch_nodes, _bid2node

            def _order_candidate_ccs():
                # todo: check how baechi calculate the peak memory: esp. temp memory
                candidate_ccs = []
                print("device, peak ops and peak ccs")
                print(self._device_id, self.peak_ops)
                print(self.peak_ccs)
                for ccs_id, peak_op_ids_in_cc in self.peak_ccs.items():
                    # peak memory contributed by peak ops
                    ops_peak_memory = 0
                    for op_id in peak_op_ids_in_cc:
                        op = self.ops_on_device[op_id]
                        ops_peak_memory += op.get_op_memory() - op.get_op_info()["persistent_memory"]
                    # memory is the sum of the persistent memory of this cc and the memory of peak ops in this cc
                    memory = self.ccs_on_device[ccs_id].get_cc_persistent_memory() + ops_peak_memory
                    candidate_ccs.append(Node(ccs_id, memory))
                candidate_ccs.sort()
                return candidate_ccs

            def _get_kept(sorted_array, used, limit, node_type, bid2node=None):
                left_quota = limit - used
                print("left quota")
                print("candidate items")
                for e in sorted_array:
                    print(e)
                nodes_to_keep = []
                while sorted_array and left_quota > 0:
                    index = bisect.bisect_right(sorted_array, Node("", left_quota))
                    print("index found:", index)
                    if index == 0:
                        break
                    else:
                        nodes_to_keep.append(sorted_array[index - 1])
                        used += sorted_array[index - 1].memory
                        left_quota = limit - used
                        item_id = sorted_array[index - 1].id
                        sorted_array.pop(index - 1)
                        if node_type == "branch":
                            for _branch_id in branches_dict[item_id]["cascade_set"]:
                                cascade_delete_branch = branches_dict[_branch_id]
                                # todo: double check if not in array is correct
                                if cascade_delete_branch in sorted_array:
                                    sorted_array.remove(bid2node[_branch_id])
                return nodes_to_keep, sorted_array, used

            """return: a new CC after remove kept ops from the orignal cc"""

            # list of (id, memory) in increasing order of memory
            ccs_nodes_to_remove = _order_candidate_ccs()
            print("ordered ccs nodes (ccs_id and ccs memory)", ccs_nodes_to_remove)
            used_memory = 0
            # todo: set as 0.8 for now, need more consideration
            #
            ccs_nodes_to_keep, ccs_nodes_to_remove, used_memory = \
                _get_kept(ccs_nodes_to_remove, used_memory, self._memory_cap, "ccs")

            # print("ccs nodes to keep: ", ccs_nodes_to_keep)
            # print("ccs nodes to remove:", ccs_nodes_to_remove)

            ccs_to_remove = {}
            for ccs_node in ccs_nodes_to_remove:
                # print("ccs_id and CC object:", ccs_node.id, self.ccs_on_device[ccs_node.id])
                ccs_to_remove[ccs_node.id] = self.ccs_on_device[ccs_node.id]

            print("device all ccs to remove:", ccs_to_remove)

            # if larger than ratio * memory cap: prefer waste of memory to avoid split cc
            ratio = 0.8
            if used_memory <= ratio * self._memory_cap:
                print("remained too little, need to split to_remove_ccs")
                # removed too much, split cc
                # ccs_to_keep = []
                # for ccs_node in ccs_nodes_to_keep:
                #     self.remove_cc(self.ccs_on_device[ccs_node.id])
                # branches_dict have more info: eg: ops in branch
                # branches is just a list of (branch_id, memory)
                branches_dict, ordered_branch_nodes, bid2node = _get_ordered_op_branches(ccs_to_remove)
                branch_nodes_to_keep, branch_nodes_to_remove, used_memory = _get_kept(ordered_branch_nodes, used_memory,
                                                                                      self._memory_cap, "branch",
                                                                                      bid2node)

                # todo: haven't done yet
                for branch_node in branch_nodes_to_remove:
                    branch = branches_dict[branch_node.id]
                    cc = self.ccs_on_device[branch["ccs_id"]]
                    new_cc = cc.remove_branch(branch)
                    ccs_to_remove[new_cc.get_ccs_id()] = new_cc
            # todo: remmber to update the device's used memory: need it to find out the available devices
            # print("all ccs to remove:", ccs_to_remove)
            return ccs_to_remove

        return _heuristic1()


class CC:
    cur_available_ccs_id = -1

    @classmethod
    # return and increase the available ccs_id.
    def get_cur_available_ccs_id(cls):
        CC.cur_available_ccs_id += 1
        return CC.cur_available_ccs_id

    def __init__(self, op_ids, device_id):
        # ops: should be a dict of OP objects, key is the op_id, val is OP object
        self.ccs_id = CC.get_cur_available_ccs_id()
        self.cc_device_id = device_id
        self.ops_in_cc = self._set_ops_in_cc(op_ids)
        self.ccs_weight = self._calculate_cc_weight()
        # self.persistent_memory, self.max_op_id = self._calculate_cc_memory()
        self.memory_simple_sum = self._sum_op_memories()
        self.co_groups_in_cc = {}
        self.topo_sorted_gname_list = []
        self.group_cuts = []

    def __lt__(self, other):
        return self.ccs_weight > other.ccs_weight

    def _set_ops_in_cc(self, op_ids):
        ops_in_cc = {}
        for op_id in op_ids:
            ops_in_cc[op_id] = OP(op_id, self.ccs_id, self.cc_device_id)
        return ops_in_cc

    def get_edge_total_tensor(self, edge):
        tensors = edge['tensor']
        tensor_size = 0
        for tensor in tensors:
            tensor_size += tensor['num_bytes']
        # print("tensor size:", tensor_size)
        return tensor_size

    def add_group_to_cc(self, group):
        self.co_groups_in_cc[group.group_name] = group

    def topo_sort_groups_old(self):
        """based on the topological order of the whole graph"""
        group_list = []
        for g_name, g_info in self.co_groups_in_cc.items():
            group_list.append((g_name, g_info.group_topo_order))
        # sort by topo order
        group_list.sort(key=lambda x: x[1])
        self.topo_sorted_gname_list = [x[0] for x in group_list]

    def topo_sort_groups(self, G):
        """try set topo_sort separately, need the orignal graph"""
        group_list = []
        subG = G.subgraph(self.co_groups_in_cc.keys())
        # for g_name, g_info in self.co_groups_in_cc.items():
        #     group_list.append((g_name, g_info.group_topo_order))
        # # sort by topo order
        # group_list.sort(key=lambda x: x[1])
        self.topo_sorted_gname_list = list(nx.topological_sort(subG))
        # self.topo_sorted_gname_list = self.dfs_topo_order(subG)

    def get_tensor_given_cut(self, index, group_graph):
        second_half = set(self.topo_sorted_gname_list[index + 1:])
        tensor = 0
        for gname in self.topo_sorted_gname_list[0:index + 1]:
            out_edges = group_graph.out_edges(gname)
            # print("gname and out edges", gname, out_edges)
            for edge in out_edges:
                out_end = edge[1]
                if out_end in second_half:
                    # todo: double check edge in group graph
                    tensor += group_graph.edges[edge]['tensor']
        return tensor

    def set_sorted_group_cuts(self, group_graph):
        ccs_id = self.ccs_id
        topo_sorted_gname_list = self.topo_sorted_gname_list
        for index in range(1, len(topo_sorted_gname_list) - 1):
            cut_info = {'ccs_id': ccs_id, 'index': index,
                        'tensor': self.get_tensor_given_cut(index, group_graph)}
            self.group_cuts.append(cut_info)
        self.group_cuts.sort(key=lambda x: x['tensor'])

    def _set_colocation_groups(self):
        groups = {}
        memories = {}
        for op_id, op in self.ops_in_cc.items():
            group = op.get_op_info()['colocation_group']
            groups[group] = groups.get(group, [])
            groups[group].append(op_id)
            memories[group] = memories.get(group, 0) + op.get_op_memory()
        co_groups = {}
        for g_name, g_ops in groups.items():
            co_groups[g_name] = Group(g_name, g_ops, memories[g_name])
        return co_groups

    def get_colocation_groups(self):
        return self.colocation_groups

    def get_cc_simple_sum_memory(self):
        return self.memory_simple_sum

    def _sum_op_memories(self):
        m = 0
        for op_id, op in self.ops_in_cc.items():
            m += op.get_op_memory()
        return m

    def _pop_ops_from_cc(self, ops_to_remove):
        for _, op in ops_to_remove.items():
            op.set_op_belong_to_ccs_id(-1)
            op.set_op_belong_to_device_id(-1)
            self.ops_in_cc.pop(op.get_op_id())
            self.persistent_memory -= op.get_op_info()["persistent_memory"]
            self.ccs_weight -= op.get_op_info()["weight"]

    def _set_ccs_id_to_ops(self):
        # print("should be dic:", type(self.ops_in_cc))
        for _, op in self.ops_in_cc.items():
            op.set_op_belong_to_ccs_id(self.ccs_id)

    def set_cc_belong_to_device_id(self, device_id):
        # print("device id:", device_id)
        self.cc_device_id = device_id
        self._set_device_id_to_ops_in_cc()

    def _set_device_id_to_ops_in_cc(self):
        # print("self.cc_device_id", self.cc_device_id)
        for _, op in self.ops_in_cc.items():
            op.set_op_belong_to_device_id(self.cc_device_id)

    def get_cc_device_id(self):
        return self.cc_device_id

    def get_ccs_id(self):
        return self.ccs_id

    def get_cc_persistent_memory(self):
        return self.persistent_memory

    def get_ops_in_cc(self):
        return self.ops_in_cc

    def _calculate_cc_weight(self):
        weight = 0
        for _, op in self.ops_in_cc.items():
            weight += op.get_op_info()["weight"]
        return weight

    def _calculate_cc_memory(self):
        max_op_memory = -1
        max_op_id = -1
        persistent_memory = 0
        for _, op in self.ops_in_cc.items():
            persistent_memory += op.get_op_info()["persistent_memory"]
            max_op_id = op.get_op_id() if op.get_op_memory() > max_op_memory else max_op_id
        if max_op_id < 0:
            print("max op negative:", self.ops_in_cc.keys())
        return persistent_memory, max_op_id

    # def _dfs(self, G, root, visited):
    #     if not root or root in visited:
    #         return visited
    #     visited.add(root)
    #     # print("edges of root:", G.edges(root))
    #     neighbors = [edge[1] for edge in list(G.edges(root))]
    #     for neigh in neighbors:
    #         self._dfs(G, neigh, visited)
    #     return visited


class OP:
    _op_graph = None

    @classmethod
    def set_op_graph(cls, op_graph):
        OP._op_graph = op_graph

    def __init__(self, op_id, ccs_id, device_id):
        self.op_id = op_id
        self.op_info = OP._op_graph.nodes[op_id]
        self.op_memory = self._get_simple_sum_memory_one_op()
        self.op_belong_to_ccs_id = ccs_id
        self.op_belong_to_device_id = device_id
        assert device_id == self.op_info['p'], \
            "when init, device id should be consistent with op's p"

    """set"""

    def set_op_belong_to_ccs_id(self, ccs_id):
        self.op_belong_to_ccs_id = ccs_id

    def set_op_belong_to_device_id(self, device_id):
        self.op_belong_to_device_id = device_id
        OP._op_graph.nodes[self.op_id]['p'] = device_id

    """get"""

    def get_op_id(self):
        return self.op_id

    def get_op_info(self):
        return self.op_info

    def get_op_memory(self):
        return self.op_memory

    def get_op_topo_order(self):
        return self.op_info["topo_order"]

    def get_op_belong_to_ccs_id(self):
        return self.op_belong_to_ccs_id

    def get_op_belong_to_device_id(self):
        return self.op_belong_to_device_id

    def _get_simple_sum_memory_one_op(self):
        memory_sum = self.op_info['temporary_memory'] + self.op_info['persistent_memory'] + sum(
            self.op_info['output_memory'])
        return memory_sum


class Group:
    def __init__(self, name, ops):
        self.group_name = name
        self.group_ops = {}
        self.group_memory = 0
        self.group_topo_order = -1
        self.group_ccs = {}
        for op in ops:
            self.add_op_to_group(op)

    def add_op_to_group(self, op):
        self.group_ops[op.get_op_id()] = op
        self.group_memory += op.get_op_memory()
        ccs_id = op.get_op_belong_to_ccs_id()
        if ccs_id in self.group_ccs:
            self.group_ccs[ccs_id].add(ccs_id)
        else:
            self.group_ccs[ccs_id] = {ccs_id}

    def set_group_topo_order(self, order):
        self.group_topo_order = order

    def get_group_topo_order(self):
        return self.group_topo_order

    def get_group_name(self):
        return self.group_name

    def get_group_ops(self):
        return self.group_ops

    def get_group_memory(self):
        return self.group_memory


class Reallocator:
    def __init__(self, op_index, op_graph, device_graph):
        print("**********initilating a reallocator*************")
        # self._placer = placer
        self.op_index = op_index
        self.index_op = self._init_index_op_dic()
        self._op_graph = op_graph
        self._device_graph = device_graph


        self.devices = self._init_devices_dic()
        OP.set_op_graph(op_graph)
        ReDevice.set_op_graph(op_graph)
        self._redevices_dic = self._init_redevice_info()
        print('what is the device graph?', self._device_graph)

        self._ccs_dic = self._init_ccs_dic()
        self._ops_dic = self._init_ops_dic()
        self._co_groups = self._init_group_dic()
        self._group_graph = self.build_group_graph()
        self._group_topo_order = self.init_group_topo_order()
        self._add_groups_to_cc()
        # added ex_type_dic for different experiments
        self.ex_type_dic = {'baechi': self.get_op_allocation,
                            'metric_balance': self.metric_balance,
                            }
        # 'diff_b_ex': self.diff_b_ex}

    """initialization"""

    def _init_index_op_dic(self):
        dic = {}
        for op_name, op_index in self.op_index.items():
            dic[op_index] = op_name
        return dic

    def _init_devices_dic(self):
        dic = {}
        for did, device in self._device_graph.nodes.items():
            dic[did] = device['name']
        return dic

    def _init_redevice_info(self):
        device2opids = {}
        for op_id, op_info in self._op_graph.nodes.items():
            device_id = op_info['p']
            device2opids[device_id] = device2opids.get(device_id, [])
            device2opids[device_id].append(op_id)

        redevices_dic = {}
        # devices_set = set(self._new_devices_dic.keys()).union(self._device_graph.nodes)
        for device_id in device2opids:  # if a new device, no ops on it, therefore use device2ops
            G = get_independent_subgraph(self._op_graph, device2opids[device_id])
            # should be a list of set of op ids
            device_ccs = list(nx.weakly_connected_components(G))
            memory_cap = self._device_graph.nodes[device_id]['memory_limit']
            redevices_dic[device_id] = ReDevice(device_id, device_ccs, memory_cap=memory_cap)
            print("this is device {}".format(device_id))
            print("the number of ccs on this device is {}".format(len(device_ccs)))
            ccs_sizes = [len(e) for e in device_ccs]
            ccs_sizes.sort()
            print("the number of ops in each ccs is:", ccs_sizes)
        return redevices_dic

    def _init_ops_dic(self):
        ops_dic = {}
        for _, cc in self._ccs_dic.items():
            ops_dic.update(cc.get_ops_in_cc())
        return ops_dic

    def _init_ccs_dic(self):
        ccs_dic = {}
        for did, redevice in self._redevices_dic.items():
            ccs_dic.update(redevice.get_ccs_on_device())
        return ccs_dic

    def _init_group_dic(self):
        # co_groups: a dic. key:group_name (from ['colocation_group']), val: a set of OP objects in this group
        co_groups = {}
        for op_id in self._op_graph.nodes:
            # for each op in self._op_graph.nodes, create a OP object
            new_op = self._ops_dic[op_id]
            # get the group info for this op
            group_name = new_op.get_op_info()['colocation_group']
            # put ops in the same group together
            if group_name in co_groups:
                co_groups[group_name].add_op_to_group(new_op)
            else:
                co_groups[group_name] = Group(group_name, [new_op])
        return co_groups

    def build_group_graph(self):
        def get_edge_total_tensor(edge):
            tensors = edge['tensor']
            tensor_size = 0
            for tensor in tensors:
                tensor_size += tensor['num_bytes']
            # print("tensor size:", tensor_size)
            return tensor_size

        # inter_group_edges: a dic. key: (from_group, to_group), val: the number of total tensor transferred
        inter_group_edges = {}
        for endpoints, edge in self._op_graph.edges.items():
            # todo: need to double check if this is the case
            # depends on the parameters, ['colocation_group']could be TensorFlow's colocation group or fused group
            from_group = self._op_graph.nodes[endpoints[0]]['colocation_group']
            to_group = self._op_graph.nodes[endpoints[1]]['colocation_group']
            # find the inter-group edges
            if not from_group == to_group:
                inter_group_edges[(from_group, to_group)] = inter_group_edges.get((from_group, to_group), 0) \
                                                            + get_edge_total_tensor(edge)

        group_graph = nx.DiGraph()
        for group, info in self._co_groups.items():
            group_graph.add_node(group, group=info)
        # group_graph.add_nodes_from(self.co_groups.items())
        for endpoints, tensor in inter_group_edges.items():
            group_graph.add_edge(*endpoints, tensor=tensor)

        print("degree for group graph")
        get_digraph_degree_freq(group_graph)
        return group_graph

    def init_group_topo_order(self):
        group_topo_orders = dfs_topo_order(self._group_graph)
        for i in range(len(group_topo_orders)):
            group = self._co_groups[group_topo_orders[i]]
            group.set_group_topo_order(i)
        return group_topo_orders

    # topo sort of a cc is called here
    def _add_groups_to_cc(self):
        for g_name, group in self._co_groups.items():
            for ccs_id in group.group_ccs:
                self._ccs_dic[ccs_id].add_group_to_cc(group)
        for ccs_id, cc in self._ccs_dic.items():
            cc.topo_sort_groups(self._group_graph)
            cc.set_sorted_group_cuts(self._group_graph)

    def _get_random_seed(self):
        seed = random.randrange(sys.maxsize)
        # seed = 1
        print("random seed is :", seed)
        return seed

    """metric balance"""
    def metric_balance(self, **kwargs):

        def _write_data_to_file(data, f_name):
            with open(f_name, "wb") as outfile:
                # "wb" argument opens the file in binary mode
                pickle.dump(data, outfile)
            print('{} has been saved'.format(f_name))

        def _initialize_data(pre_metric, need_generate_data, heuristic_type):
            def _generate_metric_data(heuristic_type):
                nonlocal coplace_groups, max_ccs_id, ccses, ccses_presum, metricvals_deviceid, ccsid_2_groups
                coplace_groups.clear()
                _max_ccs_id = -1
                # initializing coplace_groups
                # coplace_groups: {device_id: {ccs_id: [{'gname': , 'gmemory':} (topologically sorted gname)]}}
                for device_id, redevice in self._redevices_dic.items():
                    coplace_groups[device_id] = {}
                    for ccs_id, cc in redevice.ccs_on_device.items():
                        _max_ccs_id = max(_max_ccs_id, ccs_id)
                        coplace_groups[device_id][ccs_id] = []
                        for co_name in cc.topo_sorted_gname_list:
                            if heuristic_type == 'memory':
                                gname_hueristic_info = {'gname': co_name, 'gdevice_id': device_id,
                                                        'memory': cc.co_groups_in_cc[co_name].group_memory}
                            else:
                                gname_hueristic_info = {'gname': co_name, 'gdevice_id': device_id,
                                                        'op_num': len(self._co_groups[co_name].group_ops)}
                            # assert gname_gmemory['gmemory'] > 0, 'group memory is 0'
                            coplace_groups[device_id][ccs_id].append(gname_hueristic_info)

                # if baechi's placement does not use all gpus, len(self._redevices_dic.keys()) < len(self.devices.keys())
                # todo: in middle of testing

                for device_id in self.devices.keys():
                    # if cnt >= extra_device_cnt: break
                    if device_id not in self._redevices_dic.keys():
                        self._redevices_dic[device_id] = ReDevice(device_id, [])
                        coplace_groups[device_id] = {}
                        # cnt += 1

                print('devices in device graphs:', self.devices)

                update_based_on_coplace_groups()

                max_ccs_id = [_max_ccs_id]   # the current max, for each new created cc, use max+1
                
                return
            

            def _read_metric_data():
                file = 'placement_info.pickle'
                placement_info = None
                with open(file, "rb") as infile:
                    # print("reading reallocator from file:", f_name)
                    placement_info = pickle.load(infile)
                coplace_groups.update(placement_info['coplace_groups'])
                ccses.update(placement_info['ccses'])
                ccses_presum.update(placement_info['ccses_presum'])
                nonlocal metricvals_deviceid, max_ccs_id
                metricvals_deviceid = placement_info['metricvals_deviceid']
                max_ccs_id = placement_info['max_ccs_id']
                # return placement_info['coplace_groups'], placement_info['ccses'], placement_info['ccses_presum'], \
                #        placement_info['metricvals_deviceid'], placement_info['max_ccs_id']
                # return placement_info['max_ccs_id']

            if need_generate_data:
                # pre_metric = kwargs
                return _generate_metric_data(pre_metric)
            else:
                return _read_metric_data()


        def get_split_index_target(sorted_ccs_list, visited_ccs_indices, visited_metric_value, target):
            print('finding the ccs index for splitting')
            upper = target * (1 + threshold)
            lower = target * (1 - threshold)
            # new_added = []
            to_split_indices = []
            split_target = 0
            to_split_indices, split_target = [], 0
            for i in range(len(sorted_ccs_list)):
                if i not in visited_ccs_indices:
                    t = visited_metric_value + sorted_ccs_list[i]
                    # print('what is data[i], t, upper and lower', sorted_ccs_list[i], t, upper, lower, t <= upper,
                    #       t > lower)
                    if t <= upper:
                        # new_added.append(i)
                        visited_metric_value = t
                        visited_ccs_indices.append(i)
                        # cur_target -= sorted_ccs_list[i]
                        if t >= lower:
                            break
                    else:
                        to_split_indices.append(i)
                        split_target = target - visited_metric_value
                        assert split_target >= 0 and split_target <= sorted_ccs_list[i], 'split target not reasonable'
                        break
            visited_ccs_indices.sort()
            return visited_ccs_indices, visited_metric_value, to_split_indices, split_target


        def modified_subsume_approx(data, target, threshold):
            """
            data: sorted increasing ccs simple-sum memory info (not unique?)
            this is the implementation of the modified fully polynomial-tune approximation scheme of the subsum problem
            Problem: given a sorted (increasing order), unique-valued, positive integer list, a target value and a threshold
            Find: a set of elements such that their sum is within the tolerate threshold ratio within the target
            Return: the founded set or a set (potentially empty) with smaller sum and an index that needs to be split
            todo: considering move the split part out of this method
            """

            def merge_lists(L1, L2):
                """merge two sorted list into a new sorted list, removing duplication"""

                def add_to_result(val):
                    if not result or result[-1] != val:
                        # if not result or abs(result[-1] - val) > 2:  # relax the = condition since added random vals
                        result.append(val)

                result = []
                while L1 and L2:
                    l1 = L1[0]
                    l2 = L2[0]
                    if l1 <= l2:
                        add_to_result(l1)
                        L1.pop(0)
                    else:
                        add_to_result(l2)
                        L2.pop(0)
                if L1 or L2:
                    result = result + L1 if L1 else result + L2
                return result

            def trim(L, delta):
                if not L: return L
                result = [L.pop(0)]
                last = result[0]
                for e in L:
                    # if e > (1+ threshold) * target: break
                    # print('e last, and compare', e, last, )
                    if e > (1 + delta) * last:
                        result.append(e)
                        last = e
                return result

            def generate_L2(L, x, index):
                result = []
                found = False
                r = -1
                for l in L:
                    # print('sum_indices dict', sum_indices)
                    t = l + x - target
                    # print('l: {},   x: {},   index: {},    l + x - target : {},   ratio:{}'.format(l, x, index, t,
                    #                                                                                t / target))
                    if abs(l + x - target) <= target * threshold:
                        found = True
                        print('found the solution', l, x, abs(l + x - target) / target)
                        r = l + x
                        if r not in sum_indices:
                            sum_indices[r] = sum_indices[l] + [index]
                        break
                    elif l + x > (1 + threshold) * target:
                        break
                    else:
                        result.append(l + x)
                        if l + x not in sum_indices:
                            sum_indices[l + x] = sum_indices[l] + [index]
                return result, found, r

            def generate_L2_try(L, x, index):
                L2 = []
                for i in range(len(L)):
                    r = L[i] + x
                    if L[i] > target * (1 + threshold): break
                    L2.append(r)
                    if r not in sum_indices:
                        sum_indices[r] = sum_indices[L[i]] + [index]
                return L2, False, -1

            sum_indices = {0: []}
            L = [0]
            found = False
            r = -1
            assert len(data) > 0, 'len data = 0, modified_subsume_approx'
            delta = threshold / (2 * len(data))

            for i in range(len(data)):
                e = data[i]
                # print('new round and cur new element', e)
                if e > (1 + threshold) * target:
                    large_index = i
                    break
                L2, found, r = generate_L2(L, e, i)

                if found: break
                L = merge_lists(L, L2)
                # print('L after merge', L)
                L = trim(L, delta)
                # print('L after trim', L)
                upper = bisect.bisect_left(L, target * (1 + threshold))
                L = L[:upper]
                # print('L after remove', L)
                # remove large elements is done when generating L2

            cur_result_indices, cur_total_value, left_target = [], 0, 0
            # moved_value = 0  # L[-1] might be a acceptable approx.
            if found:
                cur_result_indices, left_target = sum_indices[r], 0
            else:
                cur_result_indices = sum_indices[L[-1]]
                for e in cur_result_indices:
                    cur_total_value += data[e]
                left_target = target - cur_total_value

                # split part move out, keep this method only for approx subsum purpose
                # to_move_indices, moved_memory, to_split_indices, split_target = \
                #     _get_split_index_size(to_move_indices, moved_memory, target)
            return found, cur_result_indices, cur_total_value, left_target

        
        def find_consecutive_groups(all_g_info, target, threshold):
            # todo: check: inside still use memories for name, should could handle both memory and op_num
            def presum(data_list):
                result = [0] * len(data_list)
                result[0] = data_list[0]
                for i in range(1, len(data_list)):
                    result[i] = result[i - 1] + data_list[i]
                return result

            def postsum(data_list):
                result = [0] * len(data_list)
                result[-1] = data_list[-1]
                for i in range(len(data_list) - 2, -1, -1):
                    result[i] = result[i + 1] + data_list[i]
                return result

            def cur_best_update(candidate_memory, candidate_indices, is_pre, length):
                if not cur_best['memory'] or abs(candidate_memory - target) < abs(cur_best['memory'] - target):
                    print('get a better one')
                    # candidate_indices = [start, cut_index]
                    if not is_pre:
                        left, right = candidate_indices
                        candidate_indices = [length - 1 - right + 1, length - 1 - left + 1]
                    cur_best['memory'] = candidate_memory
                    cur_best['indices'] = candidate_indices

            def check_cut(memory_sum, start, cut_index, is_pre, length):
                """
                given a memory_sum list and cut_index, check if its left and right is a qualified cut
                todo: consider about the start, where should we put it
                return: boolean, memory, indices
                boolean: if this candidate is a qualified one
                """
                print('what is the cutindex', cut_index)
                if cut_index > start:
                    if start > 0:
                        target_adjust = memory_sum[start - 1]
                    else:
                        target_adjust = 0
                    candidate_memory = memory_sum[cut_index - 1] - target_adjust
                    if candidate_memory < 0:
                        print('why candidate memeory less than 0?')
                        print(start, cut_index, is_pre)
                        exit(15)
                    candidate_indices = [start, cut_index]
                    cur_best_update(candidate_memory, candidate_indices, is_pre, length)
                    print('what is current best after lower try', cur_best['memory'], cur_best['indices'])
                    if cur_best['memory'] >= lower and cur_best['memory'] <= upper:
                        print('condition 1: larger than lower')
                        return True
                if cut_index < len(memory_sum) - 1:
                    cur_best_update(memory_sum[cut_index],
                                    [start, start + cut_index + 1], is_pre, length)
                    print('what is current best after upper try', cur_best['memory'], cur_best['indices'])
                    if cur_best and cur_best['memory'] >= lower and cur_best['memory'] <= upper:
                        print('second condition, smaller than upper')
                        return True
                return False

            presum_memories = presum(all_g_info)
            if presum_memories[-1] < target:
                return [], 0
            # assert presum_memories[-1] > target, 'the to be split cc should have larger memory than target'
            postsum_memories = postsum(all_g_info)
            postsum_memories.reverse()  # reverse it for binary search

            # simple version for now, if find one, return
            cur_best = {'memory': None, 'indices': None}
            lower = int((1 - threshold) * target)
            upper = int((1 + threshold) * target)

            # find a cut from the left side of the topological order
            # print('left try')
            cut_index = bisect.bisect_left(presum_memories, target)
            is_pre = True
            # memory_sum, start, cut_index, is_pre, length
            if cut_index > 0:
                found = check_cut(memory_sum=presum_memories, start=0, cut_index=cut_index,
                                  is_pre=is_pre, length=len(all_g_info))
                if found:
                    # print('found!')
                    return list(range(cur_best['indices'][0], cur_best['indices'][1])), cur_best['memory']

            # find a cut from the right side of the topology order
            # print('right try')
            # print('reversed postsum', postsum_memories)
            cut_index = bisect.bisect_left(postsum_memories, target)
            found = check_cut(postsum_memories, 0, cut_index,
                              False, len(all_g_info))
            if found:
                # print('found!')
                return list(range(cur_best['indices'][0], cur_best['indices'][1])), cur_best['memory']

            # get a random index and try that cut
            if len(all_g_info) > 2:
                print('trying random index')
                random.seed(self._get_random_seed())
                starts = set()
                num_groups = len(all_g_info)
                rand_cnt = min(5, num_groups - 2)
                print('the number of groups is', num_groups)
                for i in range(rand_cnt):
                    while True:
                        start = np.random.randint(1, len(all_g_info) - 1)
                        if start in starts:
                            continue
                        starts.add(start)
                        break
                    print('try random index, the rand number is ', i, start)
                    # adjust the target accordingly
                    revised_target = target + presum_memories[start - 1]
                    print('target and revised target', target, revised_target)
                    cut_index = bisect.bisect_left(presum_memories, revised_target)
                    print('the index for rand and revised target')
                    is_pre = True
                    is_pre_info = [[True, presum_memories], [False, postsum_memories]]
                    # memory_sum, start, cut_index, is_pre, length
                    for info in is_pre_info:
                        if cut_index > 0:
                            found = check_cut(memory_sum=info[1], start=start, cut_index=cut_index,
                                              is_pre=info[0], length=len(all_g_info))
                            if found:
                                # print('found!')
                                return list(range(cur_best['indices'][0], cur_best['indices'][1])), cur_best['memory']
            else:
                print('no random position can be tried')
            if cur_best['indices']:
                return list(range(cur_best['indices'][0], cur_best['indices'][1])), cur_best['memory']
            else:
                return [], 0

        def split_one_cc(to_split_ccs_id, split_target, from_device_id=None, all_g_info=None):
            """
            return: to_split_indices: [], split_val: int
            """
            nonlocal threshold, heuristic_type, coplace_groups
            if not from_device_id:
                assert all_g_info, 'no device id or all_g_info provided for splitting a cc'
            else:
                topo_ordered_group_list = coplace_groups[from_device_id][to_split_ccs_id]
                all_g_info = [e[heuristic_type] for e in topo_ordered_group_list]
            # allgs = [e['gname'] for e in topo_ordered_group_list]
            if not all_g_info:
                print('no group info in to split cc')
                return [], 0
            elif len(all_g_info) == 1:
                print('only one group in to split cc, can not split')
                return [], 0
            else:
                return find_consecutive_groups(all_g_info, split_target, threshold)

        
        def find_reallocation_group_topo(device_groups, from_device_id, to_device_id, target, threshold):
            """
            reallocate from baechi based only on topo order, no cc level, not done yet
            """
            def update_device_groups(device_groups, to_move_group_indices, from_device_id, to_device_id):
                reverse_indices = sorted(to_move_group_indices, reverse=True)
                to_move_groups = []
                for index in reverse_indices:
                    to_move_groups.append(device_groups[from_device_id][index])
                    del device_groups[from_device_id][index]
                device_groups[to_device_id] += to_move_groups
                device_groups[to_device_id].sort(key=lambda x: x['gtopo_order'])
                return to_move_groups, device_groups

            def update_metricvals_deviceid(to_move_groups, from_device_id, to_device_id):
                moved_memory = 0
                for group in to_move_groups:
                    moved_memory += group['gmemory']
                # metricvals_deviceid = [[device_simple_sum_memory, device_id]]
                metricvals_deviceid[from_device_id][0] -= moved_memory
                metricvals_deviceid[to_device_id][0] += moved_memory

            print('in find rallocation group topo for group level')
            from_device_groups = device_groups[from_device_id]

            # require, the groups list in device_groups[from_device_id] already topo-sorted
            all_g_memories = []
            for group in from_device_groups:
                all_g_memories.append(group['gmemory'])    # require topo sorted!!!
            print('what is the group memories', all_g_memories)

            # to_move_group_indices: list of the indices based on all_g_memories that need to move.
            # to_move_memory: simple-sum memory of those groups
            to_move_group_indices, to_move_memory = find_consecutive_groups(all_g_memories, target, threshold)
            to_move_groups, device_groups = update_device_groups(device_groups, to_move_group_indices, from_device_id, to_device_id)
            gnames = [e['gname'] for e in to_move_groups]
            new_op_allocations = update_op_allocation_given_gnames(gnames, from_device_id, to_device_id)
            update_metricvals_deviceid(to_move_groups, from_device_id, to_device_id)
            return new_op_allocations

        def update_op_allocation_given_gnames(gnames, from_device_id, to_device_id):
            """
            reallocator: the reallocator
            """
            new_op_allocations = {}
            for gname in gnames:
                for op_id in self._co_groups[gname].group_ops:
                    new_op_allocations[op_id] = to_device_id
            return new_op_allocations


        def get_move_dict_list(threshold):
            # todo: need to deal with unbalnced situation, i.e., over: 15 + 15 + 15 = 45: that is one device is very low
            # rebalance_metrics = [[1523, 0], [2000, 1], [654, 2], [540, 3]]
            # print("testing move dictionary")
            # rebalance_metrics = [(1609358848, 0), (1264652800, 1), (1214149888, 2), (523814912, 3)]
            print('rebalance_metrics', rebalance_metrics)
            print('ccses', ccses)
            print('ccses-presum', ccses_presum)
            print('simple-sum memory', metricvals_deviceid)
            ave, _ = np.average(rebalance_metrics, axis=0)
            ave = int(ave)
            print('average:', ave)
            # ratio = [[e[0]/ave, e[1]] for e in rebalance_metrics]
            from_hp, to_hp = [], []
            from_m_sum, to_m_sum = 0, 0
            cur_memories = [0] * len(rebalance_metrics)
            for memory, device_id in rebalance_metrics:
                cur_memories[device_id] = memory
                if memory > ave * (1 + threshold):
                    # from_hp.append([memory - ave, device_id])
                    from_hp.append([ave - memory, device_id])  # negative value, therefore max hp
                    from_m_sum += memory - ave
                elif memory < ave * (1 - threshold):
                    to_hp.append([memory - ave, device_id])  # negative value, therefore max hp
                    to_m_sum += ave - memory

            heapq.heapify(from_hp)
            heapq.heapify(to_hp)

            move_dict_list = []
            while from_hp and to_hp:
                from_r, from_device_id = heapq.heappop(from_hp)
                to_r, to_device_id = heapq.heappop(to_hp)
                from_r, to_r = abs(from_r), abs(to_r)
                assert from_device_id == metricvals_deviceid[from_device_id][1]
                # logic one is based on the from device
                logic_one = metricvals_deviceid[from_device_id][0] / rebalance_metrics[from_device_id][0]
                print('logic one for calculating reallocation dictionary', logic_one)
                # check if they both can be satisfied
                if abs(from_r - to_r) / from_r <= threshold:
                    # memory = from_r / (1 + from_r) * rebalance_metrics[from_device_id[0]]
                    to_move_amount = int(from_r * logic_one)
                    move_dict_list.append({'from': from_device_id, 'to': to_device_id, 'target': to_move_amount})
                    cur_memories[from_device_id] -= to_move_amount
                    cur_memories[to_device_id] += to_move_amount
                    continue
                # case: from_r is much larger than to_r
                if from_r > to_r:
                    # memory = to_r / (1 + from_r) * rebalance_metrics[from_device_id[0]]
                    to_move_amount = int(to_r * logic_one)
                    move_dict_list.append({'from': from_device_id, 'to': to_device_id, 'target': to_move_amount})
                    heapq.heappush(from_hp, [-1 * (from_r - to_r), from_device_id])
                    cur_memories[from_device_id] -= to_move_amount
                    cur_memories[to_device_id] += to_move_amount
                # case: from_r is smaller than to_r
                else:
                    to_move_amount = int(from_r * logic_one)
                    move_dict_list.append({'from': from_device_id, 'to': to_device_id, 'target': to_move_amount})
                    heapq.heappush(to_hp, [-1 * (to_r - from_r), to_device_id])
                    cur_memories[from_device_id] -= to_move_amount
                    cur_memories[to_device_id] += to_move_amount
            print('move dict list', move_dict_list)
            return move_dict_list

        def presum(data_list):
            result = [0] * len(data_list)
            result[0] = data_list[0]
            for i in range(1, len(data_list)):
                result[i] = result[i - 1] + data_list[i]
            return result

        def get_initial_groups_on_device_info(topo_sorted_list):
            f"""
            input: the topologically sorted group names of the reallocator: all groups
            output: a dictionary key: device id, val: list of groups {'gname', 'gmemory', 'gtopo_order'}, 
                    sorted by the gtopo_order 
            """
            device_id_2_group_info = {}
            group_2_topo_order = {}
            for i in range(len(topo_sorted_list)):
                gname = topo_sorted_list[i]
                group_2_topo_order[gname] = i

            # coplace_groups: {device_id: {ccs_id: [{'gname':, 'gmemory':} (topologically sorted gname)]}}
            for device_id, ccs_dic in coplace_groups.items():
                group_info = []
                for _, group_list in ccs_dic.items():
                    for group in group_list:
                        group_info.append({'gname': group['gname'], 'gmemory': group['gmemory'],
                                           'gtopo_order': group_2_topo_order[group['gname']]})
                group_info.sort(key=lambda x: x['gtopo_order'])
                device_id_2_group_info[device_id] = group_info
            return device_id_2_group_info

        def re_balance_group_topo(threshold):
            print('max ccs id in rebalance', max_ccs_id)
            reallocated_op_allocations = {}
            # indices = [i for i in range(len(cur_metrics))]
            device_groups = get_initial_groups_on_device_info(self._group_topo_order)

            move_dict_list = get_move_dict_list(threshold)
            for move_dict in move_dict_list:
                from_device_id, to_device_id, target = move_dict['from'], move_dict['to'], move_dict['target']
                # target = int(metricvals_deviceid[from_device_id][0] * from_ratio)
                print('move -------- from {} to {} target {}--------'.format(from_device_id, to_device_id,
                                                                             humanize_num_bytes(target)))
                sub_new_allocations = \
                    find_reallocation_group_topo(device_groups, from_device_id, to_device_id, target, threshold)

                reallocated_op_allocations.update(sub_new_allocations)
            return reallocated_op_allocations

        def update_based_on_coplace_groups():
            nonlocal ccses, ccses_presum, metricvals_deviceid
            ccses.clear()
            # ccses = {device_id: [[heuristic_metric_of_this_cc, ccs_id]]}
            # items in ccses are sorted in increasing order of heuristic_metric_of_this_cc
            for device_id, ccs_sorted_group_list in coplace_groups.items():
                ccses[device_id] = []
                metric_in_ccs = []
                for ccs_id, sorted_group_list in ccs_sorted_group_list.items():
                    cc_heuristic_metric = 0
                    for group_info in sorted_group_list:
                        cc_heuristic_metric += group_info[heuristic_type]
                    metric_in_ccs.append([cc_heuristic_metric, ccs_id])
                metric_in_ccs.sort()
                ccses[device_id] = metric_in_ccs

            ccses_presum.clear()
            for device_id, sorted_ccs_metric_info in ccses.items():
                if not sorted_ccs_metric_info:
                    ccses_presum[device_id] = [0]
                    continue
                sorted_vals = [e[0] for e in sorted_ccs_metric_info]
                ccses_presum[device_id] = presum(sorted_vals)

            metricvals_deviceid.clear()
            for i in range(len(ccses)):
                metricvals_deviceid.append([0, i])
            for i in range(len(ccses)):
                metricvals_deviceid[i] = [ccses_presum[i][-1], i]


        def re_balance(threshold, level_type='two_levels', heuristic_type='op_num'):
            # todo: have trouble with using old methods for updateing data after each run
            #  working on adding ccs_id: groups info dictionary,
            """
            based on weighed heuristic metric value, globally find combinations, no device partially reallocate notion
            """
            def get_weights():
                weights = []
                for device_id in range(len(rebalance_metrics)):
                    if metricvals_deviceid[device_id][0] == 0:
                        weights.append(0)
                    else:
                        weights.append(rebalance_metrics[device_id][0] / metricvals_deviceid[device_id][0])
                return weights

            def update_groups(weights, heuristic_type):
                for device_id, ccs_dictionary in coplace_groups.items():
                    for ccs_id, sorted_group_list in ccs_dictionary.items():
                        for group_dictionary in sorted_group_list:
                            group_dictionary[heuristic_type] *= weights[device_id]

            def all_groups_in_to_split_cc(to_split_ccs_id):
                group_metric_vals = []
                for device_id, ccs_dict in coplace_groups.items():
                    if to_split_ccs_id in ccs_dict:
                        groups = ccs_dict[to_split_ccs_id]
                        group_metric_vals = [e[heuristic_type] for e in groups]
                assert group_metric_vals, 'cannot find groups info in to split ccs'
                return group_metric_vals

            def get_ids_give_indices(given_list, indices_list):
                ccs_ids = []
                for index in indices_list:
                    ccs_ids.append(given_list[index][1])
                return ccs_ids

            def update_ccs_groups_dic_given_weights():
                for ccs_id, ccs_info in ccsid_2_groups.items():
                    weight = weights[ccs_info['device_id']]
                    for group_info in ccs_info['groups']:
                        device_id = group_info['gdevice_id']
                        group_info[heuristic_type] *= weight
                    ccs_info['ccs_metric'] *= weight

            def update_available_sorted_metric_ccsid():
                available_ccs_metric_ccsid.clear()
                for ccs_id, ccs_info in ccsid_2_groups.items():
                    available_ccs_metric_ccsid.append([ccs_info['ccs_metric'], ccs_id])
                    # if ccs_id not in scheduled_ccsids:
                available_ccs_metric_ccsid.sort()

            def create_new_ccs_from_split():
                # nonlocal max_ccs_id
                cc_info_to_split = ccsid_2_groups[to_split_ccs_id]
                new_cc = {}
                new_cc_groups = []
                new_cc_metric = 0
                for i in move_coplace_group_indices:
                    new_cc_groups.append(cc_info_to_split['groups'][i])
                    new_cc_metric += cc_info_to_split['groups'][i][heuristic_type]

                max_ccs_id[0] += 1
                new_cc['ccs_id'] = max_ccs_id[0]
                print('a new cc is created, the ccs id for the new cc is', new_cc['ccs_id'])
                new_cc['ccs_metric'] = new_cc_metric
                new_cc['device_id'] = device_id
                new_cc['groups'] = new_cc_groups
                ccsid_2_groups[new_cc['ccs_id']] = new_cc

                move_coplace_group_indices.sort(reverse=True)
                for index in move_coplace_group_indices:
                    del(ccsid_2_groups[to_split_ccs_id]['groups'][index])
                ccsid_2_groups[to_split_ccs_id]['ccs_metric'] -= new_cc_metric
                assert ccsid_2_groups[to_split_ccs_id]['ccs_metric'] > 0

                ccsid_keys = list(ccsid_2_groups.keys())
                ccsid_keys.sort()
                print('all ccs in ccsid_2_groups after creat a new cc',ccsid_keys)
                return new_cc['ccs_id']

            def update_ccs_groups_dic():
                # print('what is the ccs_ids_on_device', ccs_ids_on_device)
                for ccs_id in ccs_ids_on_device:
                    # print('what is the ccs_id', ccs_id)
                    # print('is this ccs_id in ccsid_2_groups?', ccs_id, ccs_id in ccsid_2_groups)

                    new_scheduling[ccs_id] = ccsid_2_groups.pop(ccs_id)
                    # print('waht is the newly added one', new_scheduling[ccs_id])
                    # print('new_scheduling[ccs_id]', new_scheduling[ccs_id])
                    new_scheduling[ccs_id]['device_id'] = device_id
                    # print('new_scheduling[ccs_id] after ', new_scheduling[ccs_id])

            def generate_allocation():
                op_allocations = {}
                for ccs_id, ccs_info in ccsid_2_groups.items():
                    groups = ccs_info['groups']
                    _device_id = ccs_info['device_id']
                    for group in groups:
                        gname = group['gname']
                        op_ids_in_group = self._co_groups[gname].group_ops.keys()
                        for op_id in op_ids_in_group:
                            op_allocations[op_id] = _device_id
                return op_allocations


                # for device_id, ccs_dict in coplace_groups.items():
                #     # print('what is an item in coplace groups, device_id, ccs_dict:', device_id, ccs_dict)
                #     for ccs_id, group_list in ccs_dict.items():
                #         # print('what is an item in ccs dict, ccs_id, group_list:', ccs_id, group_list)
                #         for group_info in group_list:
                #             # print('what is in the group_list', group_info)
                #             # exit(5)
                #             # print('what is th=ype of group info', type(group_info))
                #             if type(group_info) != dict:
                #                 print('not a dict!', group_info)
                #                 exit(5)
                #             # exit(5)
                #             gname = group_info['gname']
                #             for op_id in self._co_groups[gname].group_ops.keys():
                #                 op_allocations[op_id] = device_id
                # return op_allocations


            nonlocal coplace_groups, ccses, ccses_presum, metricvals_deviceid, max_ccs_id, ccsid_2_groups
            # print('max ccs id in rebalance', max_ccs_id)
            reallocated_op_allocations = {}
            ccsid_keys = list(ccsid_2_groups.keys())
            ccsid_keys.sort()
            # print('entering re_balance, what is the ccsid_2_groups', ccsid_keys)



            weights = get_weights()
            # print('weights for this round', weights)
            # update_groups(weights, heuristic_type)
            # update_based_on_coplace_groups()
            update_ccs_groups_dic_given_weights()

            ccsid_keys = list(ccsid_2_groups.keys())
            ccsid_keys.sort()
            # print('after update with weights, what is the ccsid_2_groups', ccsid_keys)

            update_based_on_coplace_groups()

            sum_peak_memories = [e[0] for e in rebalance_metrics]
            target = np.average(sum_peak_memories)

            new_scheduling = {}

            available_ccs_metric_ccsid = []


            for device_id in range(len(metricvals_deviceid)):
                # new_ccs_allocations[device_id] = set()
                ccs_ids_on_device = []
                # to_split_ccs_indices_list,  = [], 0
                to_split_ccs_indices_list, to_split_ccs_id, to_split_amount = [], -1, 0
                move_coplace_group_indices, moved_split_memory = [], 0

                if device_id == len(metricvals_deviceid) - 1:
                    # the last device, put all left on this device
                    ccs_ids_on_device = list(ccsid_2_groups.keys())
                else:

                    ccsid_keys = list(ccsid_2_groups.keys())
                    ccsid_keys.sort()
                    # print('device , what is the ccsid_2_groups', device_id, ccsid_keys)

                    update_available_sorted_metric_ccsid()
                    if not available_ccs_metric_ccsid:
                        print('no ccs left for scheduling')
                        break
                    available_sorted_metric_vals = [e[0] for e in available_ccs_metric_ccsid]
                    # print('available ccs vals and target', available_sorted_metric_vals, target)

                    found, ccs_indices_list, cur_total, left_target = \
                        modified_subsume_approx(data=available_sorted_metric_vals, target=target, threshold=threshold)
                    # print('ccs ids found after subsum', ccs_indices_list)

                    if not found:
                        # move_ccs_indices_list, to_split_ccs_indices_list, to_split_amount = find_cut(ccs_list, target, threshold,
                        #                                                                              heuristic_type)
                        ccs_indices_list, moved_total, to_split_ccs_indices_list, to_split_amount = \
                            get_split_index_target(available_sorted_metric_vals, ccs_indices_list, cur_total, target)
                        # print('ccs ids found after further check if not found', ccs_indices_list)

                    if level_type == 'two_levels' and to_split_ccs_indices_list and to_split_amount > 0:
                        # to_split_ccs_index = to_split_ccs_indices_list[0]
                        to_split_ccs_id = available_ccs_metric_ccsid[to_split_ccs_indices_list[0]][1]
                        all_g_info = all_groups_in_to_split_cc(to_split_ccs_id)

                        move_coplace_group_indices, moved_split_val = \
                            split_one_cc(to_split_ccs_id=to_split_ccs_id, split_target=to_split_amount, all_g_info=all_g_info)


                    ccs_ids_on_device = get_ids_give_indices(available_ccs_metric_ccsid, ccs_indices_list)
                    if move_coplace_group_indices:
                        ccs_ids_on_device.append(create_new_ccs_from_split())
                        # print('ccs ids found after new split cc', ccs_indices_list)

                # delete ccs in ccs_ids_on_device from ccs_to_groups and adding to new_scheduling with new device id info
                # print('device id, len of old, new', device_id, len(ccsid_2_groups), len(new_scheduling))
                update_ccs_groups_dic()
                # print("out of the method call, where did i change during iteration?")

            ccsid_2_groups = new_scheduling
            new_op_allocations = generate_allocation()
            return None, new_op_allocations


        def get_ccsid_2_groups():
            dic = {} # key: ccs_id, val: topologically sorted group infos
            for device_id, ccs_dic in coplace_groups.items():
                for cc_id, group_list in ccs_dic.items():
                    dic[cc_id] = {'groups': [], 'ccs_metric': 0, 'device_id': -1, 'ccs_id': -1}
                    dic[cc_id]['groups'] = group_list
                    ccs_metric = 0
                    for group in group_list:
                        ccs_metric += group[heuristic_type]
                    dic[cc_id]['ccs_metric'] = ccs_metric
                    dic[cc_id]['device_id'] = device_id
                    dic[cc_id]['ccs_id'] = cc_id
            return dic

        def update_co_groups_given_ccs_dict():
            coplace_groups.clear()
            for ccs_id, ccs_info in ccsid_2_groups.items():
                device_id = ccs_info['device_id']
                groups = ccs_info['groups']
                coplace_groups[device_id] = coplace_groups.get(device_id, {})
                coplace_groups[device_id][ccs_id] = groups

        ccs_id_tracker, pre_metric = {}, {}
        need_generate_data = kwargs['need_generate_data']
        level_type = 'two_levels'
        # heuristic_type = 'memory'
        heuristic_type = 'op_num'
        threshold = 0.15

        if kwargs:
            if 'need_generate_data' in kwargs:
                need_generate_data = kwargs['need_generate_data']
            if 'level_type' in kwargs:
                level_type = kwargs['level_type']
            if 'heuristic_type' in kwargs:
                heuristic_type = kwargs['heuristic_type']
            if 'threshold' in kwargs:
                threshold = kwargs['threshold']

        # prepare for reallocation.
        print('am i here in the metric balancing?')
        coplace_groups, ccses, ccses_presum, metricvals_deviceid, max_ccs_id = {}, {}, {}, [], []

        _initialize_data(pre_metric, need_generate_data, heuristic_type)

        ccsid_2_groups = get_ccsid_2_groups()

        rebalance_metrics = []
        if kwargs and kwargs['rebalance_metrics'] and kwargs['rebalance_metrics']:
            metrics = kwargs['rebalance_metrics']
            device_ids = list(range(len(metrics)))
            rebalance_metrics = list(zip(metrics, device_ids))
        else:
            print('no passed in metrics, use mericals_deviceis', metricvals_deviceid)
            for heuristic_val, device_id in metricvals_deviceid:
                rebalance_metrics.append([heuristic_val, device_id])

        print('the rebalance metrics for this round:', rebalance_metrics)

        new_allocations = {}

        if level_type in ['two_levels', 'cc_level']:
            new_allocations = re_balance(threshold, level_type, heuristic_type)
        else:
            new_allocations = re_balance_group_topo(coplace_groups, threshold, heuristic_type)



        # for new logic, directly return all op_allocations, not partial
        # new_allocations = get_all_ops_allocations(reallocated_op_allocations)
        update_co_groups_given_ccs_dict()
        update_based_on_coplace_groups()

        filename = 'placement_info.pickle'
        placement_info = {'coplace_groups': coplace_groups, 'ccses': ccses, 'ccses_presum': ccses_presum,
                          'metricvals_deviceid': metricvals_deviceid, 'max_ccs_id': max_ccs_id}
        _write_data_to_file(placement_info, filename)
        print('placement info has been written to', filename)

        return new_allocations

    def _get_most_ops_gs_ccsid(self):
        # will return the ccs id with most ops/ goups
        cur_op_max = 0
        cur_g_max = 0
        cur_op_id = -1
        cur_g_id = -1
        for ccs_id, cc in self._ccs_dic:
            if cur_op_max < len(cc.ops_in_cc):
                cur_op_max = len(cc.ops_in_cc)
                cur_op_id = ccs_id
            if cur_g_max < len(cc.co_groups_in_cc):
                cur_g_max = len(cc.co_groups_in_cc)
                cur_g_id = ccs_id
        return cur_op_id, cur_g_id

    def _get_new_op_allocations(self, op_ids, device_id):
        # return a dict, allocate ops in op_ids on device_id
        new_allocations = {}
        for op_id in op_ids:
            new_allocations[op_id] = device_id
        return new_allocations

    def _get_sorted_ccs_ops_gs_numbers(self):
        # return a list of (ccs_id, # of ops) in decreasing order of # of ops
        sorted_ccs_ops = []
        sorted_ccs_gs = []
        for ccs_id, cc in self._ccs_dic.items():
            if len(cc.ops_in_cc) > 1:
                sorted_ccs_ops.append([ccs_id, len(cc.ops_in_cc)])
            if len(cc.co_groups_in_cc) > 1:
                sorted_ccs_gs.append([ccs_id, len(cc.co_groups_in_cc)])
        sorted_ccs_ops.sort(key=lambda x: x[1], reverse=True)
        sorted_ccs_gs.sort(key=lambda x: x[1], reverse=True)
        return sorted_ccs_ops, sorted_ccs_gs

    def _get_different_rand_device(self, ccs_id=None):
        device_ids = list(self.devices.keys())
        if not ccs_id:
            old_device_id = random.choice(device_ids)
        else:
            old_device_id = self._ccs_dic[ccs_id].cc_device_id
        new_device_id = old_device_id
        while new_device_id == old_device_id:
            new_device_id = random.choice(device_ids)
        return old_device_id, new_device_id

    def _get_op_ids_from_some_cc_some_number(self, re_type):
        # some cc: can be random or the largest, given...
        # number of ops: can be random or given
        sorted_ccs_ops, sorted_ccs_gs = self._get_sorted_ccs_ops_gs_numbers()
        sorted_list = None
        if re_type == "group":
            sorted_list = sorted_ccs_gs
        else:
            sorted_list = sorted_ccs_ops
        # index = random.randint(1, len(sorted_list) - 1)  #randomly choose a cc
        index = 0  # the largest cc
        ccs_id, numbers = sorted_list[index]
        cc = self._ccs_dic[ccs_id]
        # n = random.randint(numbers//2, numbers - 1)
        n = numbers // 2
        if re_type == "group":
            target_list = list(cc.co_groups_in_cc.keys())
        else:
            target_list = list(cc.ops_in_cc.keys())
        units_to_remove = random.sample(target_list, n)
        op_ids = []
        if re_type == "group":
            for gname in units_to_remove:
                op_ids += list(self._co_groups[gname].group_ops.keys())
        else:
            op_ids = units_to_remove
        return ccs_id, op_ids, n

    def _assign_op_allocation_dic(self, op_ids, device_id):
        dic = {}
        for op_id in op_ids:
            dic[op_id] = device_id
        return dic

    def _equal_topo_group_indexes_for_each_device(self):
        # return: a dict of {"device_id: [[start_index, end_index]]}
        # the index is for the group_topo_order list
        n_groups = len(self._group_topo_order)
        n_devices = len(self.devices)
        base_n = int(n_groups / n_devices)
        device_groups_topo_indexes = {}
        device_list = list(self.devices.keys())
        for i in range(len(self.devices)):
            indexes = [i * base_n, (i + 1) * base_n]
            if i == len(self.devices) - 1:
                indexes[1] = n_groups
            device_groups_topo_indexes[device_list[i]] = []
            device_groups_topo_indexes[device_list[i]].append(indexes)
        return device_groups_topo_indexes

    def _allocation_op_given_topo_groups_indexes(self, device_groups_topo_indexes):
        new_op_allocations = {}
        for device_id, index_intervals in device_groups_topo_indexes.items():
            for interval in index_intervals:
                start_index, end_index = interval
                for index in range(start_index, end_index):
                    gname = self._group_topo_order[index]
                    for op_id in self._co_groups[gname].group_ops:
                        new_op_allocations[op_id] = device_id
        return new_op_allocations

    def _group_assign_to_op_assign(self, group_assign):
        # given group_assign dic (key: gname, value: device_id)
        # return op_assign_dic (key: op_id, value: device_id)
        op_assign_dic = {}
        for g_name, d_id in group_assign.items():
            for op_id in self._co_groups[g_name].group_ops:
                op_assign_dic[op_id] = d_id
        return op_assign_dic


        # print("group_device_assign_dic:", group_device_assign_dic)
        # print("device to slots:", device_to_slots_dic)
        g_ids = {}
        for i in range(len(self._group_topo_order)):
            g_ids[self._group_topo_order[i]] = i

        cost = np.ones((len(group_device_assign_dic), len(group_device_assign_dic)))
        for g_name, d_id in group_device_assign_dic.items():
            g_id = g_ids[g_name]
            for j in device_to_slots_dic[d_id]:
                cost[g_id][j] = 0
        return cost

    """interface"""

    def get_op_allocation(self):
        # this method just returns the op allocations in reallocator
        op_allocations = {}
        for op_id, op_object in self._ops_dic.items():
            op_allocations.update({op_id: op_object.get_op_belong_to_device_id()})
        # todo: about the return value: make it two to be consistent with other methods, may need to change in the future
        return None, op_allocations

    def get_redevice_dic(self):
        return self._redevices_dic

    def get_ccs_dic(self):
        return self._ccs_dic

    def get_co_groups_dic(self):
        return self._co_groups

    def get_ops_dic(self):
        return self._ops_dic

    def get_op_graph(self):
        return self._op_graph

    def get_group_graph(self):
        return self.group_graph

        