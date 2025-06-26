import numpy as np
from itertools import permutations, combinations
from scipy import sparse
from scipy.sparse import spmatrix
from util.net import get_pos_from_id, sp_2, sp_3
from util.data import getElites, merge_lils
from util import distance
from typing import List, Optional

from joblib import Parallel, delayed

exc_perc = 0.8
sparsity = 0.01
dist_mult = 1

add_edge = 0.01
del_edge = 0.05
chg_edge = 0.1
chg_inhb = 0.05
chg_vthr = 0.1
p_random_candidate = 0.01

param_range = [1, 127]
v_reset_range = [0, 0]
v_leak_range = [0, 127]
tau_syn_range = [10, 1000]
tau_mem_range = [10, 1000]

WORKERS = 0

p_rand = 1

eps = 0.1


class Gene:
    """
    Represents the genetic makeup of a single neural network.

    This class holds all the parameters that define the structure and behavior
    of a network, including neuron properties and synaptic weights.
    """

    def __init__(
        self,
        inhibitions,
        thresholds,
        v_reset,
        v_leak,
        tau_syn_inv,
        tau_mem_inv,
        synapses,
        delays,
        inputs,
        inp_delays,
        input_inhibitions,
        outputs,
        out_delays,
    ):
        self.inhibitions = inhibitions
        self.thresholds = thresholds
        self.v_reset = v_reset
        self.v_leak = v_leak
        self.tau_syn_inv = tau_syn_inv
        self.tau_mem_inv = tau_mem_inv

        self.synapses = synapses
        self.delays = delays
        self.inputs = inputs
        self.inp_delays = inp_delays
        self.input_inhibitions = input_inhibitions
        self.outputs = outputs
        self.out_delays = out_delays


def merge_genes(genes):
    # ensure genes is a list
    if not isinstance(genes, list):
        genes = [genes]

    inhb = np.array([g.inhibitions for g in genes]).flatten()
    vthr = np.array([g.thresholds for g in genes]).flatten()
    vres = np.array([g.v_reset for g in genes]).flatten()
    vleak = np.array([g.v_leak for g in genes]).flatten()
    tsi = np.array([g.tau_syn_inv for g in genes]).flatten()
    tmi = np.array([g.tau_mem_inv for g in genes]).flatten()
    syn = merge_lils([g.synapses for g in genes])
    delay = merge_lils([g.delays for g in genes])
    inp = merge_lils([g.inputs for g in genes])
    inp_delay = merge_lils([g.inp_delays for g in genes])
    inp_inhb = np.array([g.input_inhibitions for g in genes]).flatten()
    out = merge_lils([g.outputs for g in genes])
    out_delay = merge_lils([g.out_delays for g in genes])

    return Gene(
        inhb,
        vthr,
        vres,
        vleak,
        tsi,
        tmi,
        syn,
        delay,
        inp,
        inp_delay,
        inp_inhb,
        out,
        out_delay,
    )


class Evolution:
    """
    Manages the evolutionary process for a population of neural networks.

    This class handles the initialization of a gene pool, selection,
    crossover, and mutation operations to evolve networks based on their
    performance on a given task.
    """

    def __init__(
        self,
        net_size,
        pool_size,
        input_neurons,
        output_neurons,
        f1,
        f2,
        f3,
        max_vthr,
        prune_unconnected,
        evolution_method,
        spatial
    ):
        self.net_size = np.array(net_size)  # e.g. [3, 3, 3]
        self.vthr_range = [1, 10]
        self.vthr_range[1] = max_vthr
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.spatial = spatial
        self.neurons = np.array(net_size).prod()
        self.distance_mat = self.compute_dist_mat(distance.euclidian)
        self.inp_distance_mat = self.compute_inp_dist_mat(distance.euclidian)
        self.out_distance_mat = self.compute_out_dist_mat(distance.euclidian)
        self.synapse_prob_func = sp_2
        self.rec_prob = 0.1
        self.pool_size = pool_size
        self.evolution_method = evolution_method
        self.pool = [self.get_random_net() for i in range(pool_size)]
        self.m = int(np.sqrt(pool_size))
        self.comb_m = list(combinations(range(self.m), 2))
        self.offs_num = int(int((int(self.pool_size * 0.9) - 2 * self.m) / 2) * 2)
        self.offs_mul = int(np.ceil(self.offs_num / len(self.comb_m)) / 2)
        self.f1_size = f1
        self.f2_size = f2
        self.f3_size = f3
        self.prune_unconnected = prune_unconnected
        self.map_elites: List[List[List[Optional[Gene]]]] = [
            [[None for _ in range(f3)] for _ in range(f2)] for _ in range(f1)
        ]
        self.map_scores = [
            [[-np.inf for _ in range(f3)] for _ in range(f2)] for _ in range(f1)
        ]
        self.scores: np.ndarray = np.array([])
        self.elites: np.ndarray = np.array([])

    def update_pool(self):
        """
        Updates the gene pool for the next generation.

        This method generates a new pool of individuals by performing crossover
        and mutation on candidates selected from the current elites. A portion
        of the new pool is also filled with randomly generated individuals to
        maintain genetic diversity.
        """
        self.pool = []
        # determine non-empty f1, f2 cells
        f1s, f2s, f3s = np.nonzero(np.array(self.map_scores) != -np.inf)
        candidate_cells = list(zip(f1s, f2s, f3s))
        if len(candidate_cells) != 0:
            while len(self.pool) < int(self.pool_size * p_rand):
                # Candidate 1
                is_cand1_random = np.random.random() < p_random_candidate
                if is_cand1_random:
                    cand1 = self.get_random_net()
                else:
                    f1_1, f2_1, f3_1 = candidate_cells[
                        np.random.randint(0, len(candidate_cells))
                    ]
                    cand1 = self.map_elites[f1_1][f2_1][f3_1]

                # Candidate 2
                if np.random.random() < p_random_candidate:
                    cand2 = self.get_random_net()
                elif is_cand1_random:
                    f1_2, f2_2, f3_2 = candidate_cells[
                        np.random.randint(0, len(candidate_cells))
                    ]
                    cand2 = self.map_elites[f1_2][f2_2][f3_2]
                elif len(candidate_cells) > 1:
                    candidate_cells_2 = candidate_cells.copy()
                    candidate_cells_2.remove((f1_1, f2_1, f3_1))
                    f1_2, f2_2, f3_2 = candidate_cells_2[
                        np.random.randint(0, len(candidate_cells_2))
                    ]
                    cand2 = self.map_elites[f1_2][f2_2][f3_2]
                else:
                    cand2 = self.get_random_net()

                # Random choice of crossover or mutation
                if np.random.random() < 0.5:
                    o1, o2 = self.crossover(cand1, cand2)
                    self.pool.extend([self.mutation(o1), self.mutation(o2)])

                else:
                    self.pool.append(self.mutation(cand1))
                    self.pool.append(self.mutation(cand2))
        else:
            while len(self.pool) < int(self.pool_size * p_rand):
                # Candidate 1
                cand1 = self.get_random_net()

                # Candidate 2
                cand2 = self.get_random_net()

                # Random choice of crossover or mutation
                if np.random.random() < 0.5:
                    o1, o2 = self.crossover(cand1, cand2)
                    self.pool.extend([self.mutation(o1), self.mutation(o2)])
                else:
                    self.pool.append(self.mutation(cand1))
                    self.pool.append(self.mutation(cand2))

            if len(self.pool) > int(self.pool_size * p_rand):
                self.pool = self.pool[: int(self.pool_size * p_rand)]

        self.pool.extend(
            [self.get_random_net() for i in range(self.pool_size - len(self.pool))]
        )

    def update_elites(self):
        """
        Updates the list of elites from the map_elites archive.

        This method flattens the multi-dimensional map of elites and their
        scores into simple lists, filtering out any empty cells.
        """
        elites_list = [
            [
                [self.map_elites[f1][f2][f3] for f3 in range(self.f3_size)]
                for f2 in range(self.f2_size)
            ]
            for f1 in range(self.f1_size)
        ]
        scores_list = [
            [
                [self.map_scores[f1][f2][f3] for f3 in range(self.f3_size)]
                for f2 in range(self.f2_size)
            ]
            for f1 in range(self.f1_size)
        ]
        # flatten list of lists of lists
        flat_elites = np.array(elites_list).flatten()
        flat_scores = np.array(scores_list).flatten()

        # remove None values
        self.elites = np.array([e for e in flat_elites if e is not None])
        self.scores = np.array([s for s in flat_scores if s != -np.inf])

    def update_params_map_elites(self, rewards, ft_desc):
        """
        Updates the MAP-Elites archive with the latest results.

        For each individual in the pool, this method identifies the corresponding
        cell in the feature map and updates it if the new individual has a
        higher score.

        Args:
            rewards (np.ndarray): The rewards for each individual in the pool.
            ft_desc (np.ndarray): The feature descriptors for each individual.
        """
        for i, p in enumerate(self.pool):
            f1 = int(ft_desc[0][i] - 1)
            assert ft_desc[1][i] < 1
            f2 = int(ft_desc[1][i] * (self.f2_size))
            f3 = int(ft_desc[2][i] * (self.f3_size))
            if rewards[i] > self.map_scores[f1][f2][f3] and f1 != -1:
                self.map_scores[f1][f2][f3] = rewards[i]
                self.map_elites[f1][f2][f3] = p

        self.update_elites()
        self.update_pool()

    def update_params_classic(self, rewards, ft_desc):
        """
        Updates the population using a classic evolutionary algorithm approach.

        This method selects the top individuals based on a weighted combination
        of reward and feature descriptors, then generates offspring through
        crossover and mutation.

        Args:
            rewards (np.ndarray): The rewards for each individual in the pool.
            ft_desc (np.ndarray): The feature descriptors for each individual.
        """
        scores = [rewards]
        for f in ft_desc:
            scores.append(f)
        ascending = [False] * len(
            scores
        )  # indicates whether lower value means better score
        ascending[-1] = (
            True  # sparsity should be ascending as the value tracks the number of used synapses, i.e. lower is better
        )
        weights = [1.0] * len(
            scores
        )  # weights given for each reward when combining them
        weights[0] = 100.0  # reward is the most important
        weights[-1] = 10.0  # sparsity is the second most important
        weights[1] = 1.0  # f1 (number of used joints) is the third most important
        weights[2] = 0.1  # f2 (degree of joint usage) is the least important

        idx, _ = getElites(scores, ascending, weights)

        self.scores = rewards[idx]
        self.elites = np.array([self.pool[i] for i in idx])

        best_m = [self.pool[i] for i in idx[: self.m]]
        best_m_mutated = [self.mutation(self.pool[i]) for i in idx[: self.m]]
        offspring = []
        for i1, i2 in (self.comb_m * self.offs_mul)[: int(self.offs_num / 2)]:
            o1, o2 = self.crossover(best_m[i1], best_m[i2])
            offspring.append(self.mutation(o1))
            offspring.append(self.mutation(o2))

        rand_nets = [
            self.get_random_net()
            for i in range(self.pool_size - 2 * len(best_m) - len(offspring))
        ]

        self.pool = best_m + best_m_mutated + offspring + rand_nets

    def update_params(self, rewards, ft_desc):
        """
        Updates the population based on the chosen evolution method.

        Args:
            rewards (np.ndarray): The rewards for each individual in the pool.
            ft_desc (np.ndarray): The feature descriptors for each individual.
        """
        if self.evolution_method == "map_elites":
            self.update_params_map_elites(rewards, ft_desc)
        else:
            self.update_params_classic(rewards, ft_desc)

    def get_delay(self, x: spmatrix, distance_mat):
        """
        Calculates the delay for synapses based on their distance.

        Args:
            x (spmatrix): The synapse matrix.
            distance_mat (np.ndarray): The matrix of distances between neurons.

        Returns:
            sparse.lil_matrix: A matrix of synaptic delays.
        """
        x = x.toarray()[: distance_mat.shape[0], : distance_mat.shape[1]]
        return sparse.csr_matrix(
            (x.astype(bool).astype(int) * sp_3(distance_mat, self.spatial))
        ).tolil()

    def _compute_dist_mat(self, f_neurons, t_neurons, get_dist):
        workers = min(f_neurons, t_neurons) if WORKERS == None else WORKERS
        while workers > 64 and workers % 2 == 0:
            workers = int(workers / 2)
        if workers > 1:
            results = Parallel(n_jobs=-1)(
                delayed(get_dist)(l)
                for l in np.split(
                    np.array(
                        [(i, j) for i in range(t_neurons) for j in range(f_neurons)]
                    ),
                    workers,
                )
            )
            dists = []
            if results:
                for res in results:
                    dists.extend(res)
        else:
            dists = get_dist(
                [(i, j) for i in range(t_neurons) for j in range(f_neurons)]
            )
        return np.array(dists).reshape(t_neurons, f_neurons)

    def compute_dist_mat(self, dist_func):
        """
        Computes the distance matrix for recurrent connections.

        Args:
            dist_func (function): The function to use for calculating distance.

        Returns:
            np.ndarray: The computed distance matrix.
        """
        f_neurons = self.neurons
        t_neurons = self.neurons

        def get_dist(l):
            return [_dist(n1, n2) for n1, n2 in l]

        def _dist(n1, n2):
            pos1 = get_pos_from_id(n1, self.net_size)
            pos2 = get_pos_from_id(n2, self.net_size)
            return dist_func(pos1, pos2) * dist_mult

        distance_mat = self._compute_dist_mat(f_neurons, t_neurons, get_dist)
        return distance_mat

    def compute_inp_dist_mat(self, dist_func):
        """
        Computes the distance matrix for input connections.

        Args:
            dist_func (function): The function to use for calculating distance.

        Returns:
            np.ndarray: The computed distance matrix.
        """
        f_neurons = self.input_neurons
        t_neurons = self.neurons

        def get_dist(l):
            return [_dist(n1, n2) for n1, n2 in l]

        def _dist(n1, n2):
            pos1 = get_pos_from_id(n1, self.net_size)
            pos2 = np.copy(pos1)
            pos2[0] -= 1
            return dist_func(pos1, pos2) * dist_mult

        distance_mat = self._compute_dist_mat(f_neurons, t_neurons, get_dist)
        return distance_mat

    def compute_out_dist_mat(self, dist_func):
        """
        Computes the distance matrix for output connections.

        Args:
            dist_func (function): The function to use for calculating distance.

        Returns:
            np.ndarray: The computed distance matrix.
        """
        f_neurons = self.neurons
        t_neurons = self.output_neurons

        def get_dist(l):
            return [_dist(n1, n2) for n1, n2 in l]

        def _dist(n1, n2):
            pos2 = get_pos_from_id(n2, self.net_size)
            pos1 = np.copy(pos2)
            #pos1[0] = self.net_size[0] # place them at the end of the x axis
            pos1[0] -= 1 # place them at distance 1
            return dist_func(pos1, pos2) * dist_mult

        distance_mat = self._compute_dist_mat(f_neurons, t_neurons, get_dist)
        return distance_mat

    def prune(self, input_weights, syn, output_weights):
        """
        Prunes the network to remove unconnected neurons and isolated loops.

        Args:
            input_weights (np.ndarray): The input weight matrix.
            syn (np.ndarray): The recurrent synapse matrix.
            output_weights (np.ndarray): The output weight matrix.

        Returns:
            tuple: The pruned synapse, input, and output weight matrices.
        """
        if self.prune_unconnected:
            changed = True
            while changed:
                changed = False
                old_syn = np.copy(syn)
                old_inp = np.copy(input_weights)
                old_out = np.copy(output_weights)
                syn = self.prune_isolated_loops(input_weights, syn, output_weights)
                syn, input_weights = self.prune_nooutputs(
                    input_weights, syn, output_weights
                )
                syn, output_weights = self.prune_leftouts(
                    input_weights, syn, output_weights
                )
                changed = (
                    changed
                    or not np.array_equal(old_syn, syn)
                    or not np.array_equal(old_inp, input_weights)
                    or not np.array_equal(old_out, output_weights)
                )
        return syn, input_weights, output_weights

    def prune_nooutputs(self, input_weights, syn, output_weights):
        """
        Removes neurons that have no outgoing connections.

        Args:
            input_weights (np.ndarray): The input weight matrix.
            syn (np.ndarray): The recurrent synapse matrix.
            output_weights (np.ndarray): The output weight matrix.

        Returns:
            tuple: The pruned synapse and input weight matrices.
        """
        # Remove neurons with no output other than to itself
        has_changed = True

        while has_changed:
            hn_outputs = output_weights.astype(bool).astype(int).sum(axis=0) + (
                syn.astype(bool).astype(int)
                * (np.ones_like(syn) - np.identity(syn.shape[0]))
            ).sum(axis=0)
            no_output_idxs = np.where(hn_outputs == 0)[0]
            old_syn = np.copy(syn)
            old_inp = np.copy(input_weights)
            old_syn[no_output_idxs, :] = 0
            old_inp[no_output_idxs, :] = 0

            if np.array_equal(old_syn, syn) and np.array_equal(old_inp, input_weights):
                has_changed = False
            else:
                syn = old_syn
                input_weights = old_inp

        return syn, input_weights

    def prune_leftouts(self, input_weights, syn, output_weights):
        """
        Removes neurons that have no incoming connections.

        Args:
            input_weights (np.ndarray): The input weight matrix.
            syn (np.ndarray): The recurrent synapse matrix.
            output_weights (np.ndarray): The output weight matrix.

        Returns:
            tuple: The pruned synapse and output weight matrices.
        """
        # Remove neurons with no input other than from itself
        has_changed = True

        while has_changed:
            hn_inputs = input_weights.astype(bool).astype(int).sum(axis=1) + (
                syn.astype(bool).astype(int)
                * (np.ones_like(syn) - np.identity(syn.shape[0]))
            ).sum(axis=1)
            no_input_idxs = np.where(hn_inputs == 0)
            old_syn = np.copy(syn)
            old_out = np.copy(output_weights)
            old_syn[:, no_input_idxs] = 0
            old_out[:, no_input_idxs] = 0

            if np.array_equal(old_syn, syn) and np.array_equal(old_out, output_weights):
                has_changed = False
            else:
                syn = old_syn
                output_weights = old_out

        return syn, output_weights

    def prune_isolated_loops(self, input_weights, syn, output_weights):
        """
        Prunes isolated loops of connections within the hidden layer.

        Args:
            input_weights (np.ndarray): The input weight matrix.
            syn (np.ndarray): The recurrent synapse matrix.
            output_weights (np.ndarray): The output weight matrix.

        Returns:
            np.ndarray: The pruned recurrent synapse matrix.
        """
        # returns list of tuples, where each tuple contains the indices of neurons in an isolated loop
        isolated_loops = self.identify_isolated_loops(
            input_weights, syn, output_weights
        )
        for loop in isolated_loops:
            for k in range(len(loop)):
                i, j = loop[k], loop[(k + 1) % len(loop)]
                syn[i, j] = 0
                syn[j, i] = 0
        return syn

    def identify_isolated_loops(self, input_weights, syn, output_weights):
        """
        Identifies unconnected isolated loops in the hidden connections.

        Args:
        input_weights: numpy array of shape (n_hidden, n_input)
        syn: numpy array of shape (n_hidden, n_hidden)
        output_weights: numpy array of shape (n_output, n_hidden)

        Returns:
        list of tuples, where each tuple contains the indices of neurons in an isolated loop
        """
        n_hidden = syn.shape[0]

        # Identify neurons with inputs only from the hidden layer
        has_input = ~np.any(input_weights, axis=1) & np.any(syn, axis=1)
        # Identify neurons with outputs only to the hidden layer
        has_output = ~np.any(output_weights, axis=0) & np.any(syn, axis=0)
        # Identify neurons with both inputs and outputs
        isolated = has_input & has_output

        isolated_loops = []
        visited = set()

        for i in range(n_hidden):
            if isolated[i] and i not in visited:
                loop = self._dfs_loop(i, syn, isolated, visited)
                if loop:
                    isolated_loops.append(tuple(loop))

        return isolated_loops

    def _dfs_loop(self, start, syn, isolated, visited):
        """
        Depth-first search to find an isolated loop starting from a given neuron.
        """
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for neighbor in np.where(syn[node] != 0)[0]:
                if neighbor == start and len(path) > 1:
                    return path
                if isolated[neighbor] and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))
        return None

    def get_random_net(self):
        """
        Generates a new random network gene.

        Returns:
            Gene: A new Gene object with randomized parameters.
        """
        syn = (
            np.random.random(self.distance_mat.shape)
            < self.synapse_prob_func(self.distance_mat, self.spatial)
        ).astype(int)
        syn = syn * (
            np.random.rand(*self.distance_mat.shape) * (param_range[1] - eps) + eps
        )

        inhb = (np.random.random(self.neurons) > exc_perc).astype(int) * -2 + 1
        vthr = np.random.rand(self.neurons) * (self.vthr_range[1] - eps) + eps
        vres = (
            np.random.rand(self.neurons) * (v_reset_range[1] - v_reset_range[0])
            + v_reset_range[0]
        )
        vleak = (
            np.random.rand(self.neurons) * (v_leak_range[1] - v_leak_range[0])
            + v_leak_range[0]
        )
        tau_syn = (
            np.random.rand(self.neurons) * (tau_syn_range[1] - tau_syn_range[0])
            + tau_syn_range[0]
        )
        tau_mem = (
            np.random.rand(self.neurons) * (tau_mem_range[1] - tau_mem_range[0])
            + tau_mem_range[0]
        )

        input_weights = (
            np.random.random(self.inp_distance_mat.shape)
            < self.synapse_prob_func(self.inp_distance_mat, self.spatial)
        ).astype(int)

        input_weights = input_weights * (
            np.random.rand(self.neurons, self.input_neurons) * (param_range[1] - eps)
            + eps
        )

        rand_arr = np.random.random(self.input_neurons)
        bool_arr = rand_arr > exc_perc
        input_inhib = bool_arr.astype(int) * -2 + 1

        output_weights = (
            np.random.random(self.out_distance_mat.shape)
            < self.synapse_prob_func(self.out_distance_mat, self.spatial)
        ).astype(int)

        output_weights = output_weights * (
            np.random.rand(self.output_neurons, self.neurons) * (param_range[1] - eps)
            + eps
        )

        # syn, input_weights, output_weights = self.prune(input_weights, syn, output_weights)

        syn = sparse.csr_matrix(syn).tolil()
        delay = self.get_delay(syn, self.distance_mat)
        input_weights = sparse.csr_matrix(input_weights).tolil()
        inp_delay = self.get_delay(input_weights, self.inp_distance_mat)
        output_weights = sparse.csr_matrix(output_weights).tolil()
        out_delay = self.get_delay(output_weights, self.out_distance_mat)

        gene = Gene(
            inhb,
            vthr,
            vres,
            vleak,
            tau_syn,
            tau_mem,
            syn,
            delay,
            input_weights,
            inp_delay,
            input_inhib,
            output_weights,
            out_delay,
        )

        return gene

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent genes to create two offspring.

        Args:
            parent1 (Gene): The first parent gene.
            parent2 (Gene): The second parent gene.

        Returns:
            tuple: A tuple containing the two new offspring genes.
        """
        inhb1, vthr1, vres1, vleak1, tsi1, tmi1 = (
            parent1.inhibitions,
            parent1.thresholds,
            parent1.v_reset,
            parent1.v_leak,
            parent1.tau_syn_inv,
            parent1.tau_mem_inv,
        )
        inhb2, vthr2, vres2, vleak2, tsi2, tmi2 = (
            parent2.inhibitions,
            parent2.thresholds,
            parent2.v_reset,
            parent2.v_leak,
            parent2.tau_syn_inv,
            parent2.tau_mem_inv,
        )
        idxs = np.array(range(0, np.prod(inhb1.shape[-1])))
        np.random.shuffle(idxs)
        half1 = idxs[: int(len(idxs) / 2)]
        half2 = idxs[int(len(idxs) / 2) :]
        o_inhb1 = np.copy(inhb1)
        o_inhb2 = np.copy(inhb1)
        o_inhb1[half1] = np.copy(inhb2[half1])
        o_inhb2[half2] = np.copy(inhb2[half2])
        o_vthr1 = np.copy(vthr1)
        o_vthr2 = np.copy(vthr1)
        o_vthr1[half1] = np.copy(vthr2[half1])
        o_vthr2[half2] = np.copy(vthr2[half2])
        o_vres1 = np.copy(vres1)
        o_vres2 = np.copy(vres1)
        o_vres1[half1] = np.copy(vres2[half1])
        o_vres2[half2] = np.copy(vres2[half2])
        o_vleak1 = np.copy(vleak1)
        o_vleak2 = np.copy(vleak1)
        o_vleak1[half1] = np.copy(vleak2[half1])
        o_vleak2[half2] = np.copy(vleak2[half2])
        o_tsi1 = np.copy(tsi1)
        o_tsi2 = np.copy(tsi1)
        o_tsi1[half1] = np.copy(tsi2[half1])
        o_tsi2[half2] = np.copy(tsi2[half2])
        o_tmi1 = np.copy(tmi1)
        o_tmi2 = np.copy(tmi1)
        o_tmi1[half1] = np.copy(tmi2[half1])
        o_tmi2[half2] = np.copy(tmi2[half2])

        syn1 = parent1.synapses.toarray()
        syn2 = parent2.synapses.toarray()
        o_syn1 = np.copy(syn1)
        o_syn2 = np.copy(syn1)
        o_syn1[:, half1] = np.copy(syn2[:, half1])
        o_syn2[:, half2] = np.copy(syn2[:, half2])

        out1 = parent1.outputs.toarray()
        out2 = parent2.outputs.toarray()
        o_out1 = np.copy(out1)
        o_out2 = np.copy(out1)
        o_out1[:, half1] = np.copy(out2[:, half1])
        o_out2[:, half2] = np.copy(out2[:, half2])

        inp1 = parent1.inputs.toarray()
        inp2 = parent2.inputs.toarray()
        idxs = np.array(range(0, np.prod(inp1.shape[-1])))
        np.random.shuffle(idxs)
        half1 = idxs[: int(len(idxs) / 2)]
        half2 = idxs[int(len(idxs) / 2) :]
        o_inp1 = np.copy(inp1)
        o_inp2 = np.copy(inp1)
        o_inp1[:, half1] = np.copy(inp2[:, half1])
        o_inp2[:, half2] = np.copy(inp2[:, half2])

        inp_inhb1 = parent1.input_inhibitions
        inp_inhb2 = parent2.input_inhibitions
        o_inp_inhb1 = np.copy(inp_inhb1)
        o_inp_inhb2 = np.copy(inp_inhb1)
        o_inp_inhb1[half1] = np.copy(inp_inhb2[half1])
        o_inp_inhb2[half2] = np.copy(inp_inhb2[half2])

        # prune the networks to remove isolated loops, unconnected neurons, etc.
        o_syn1, o_inp1, o_out1 = self.prune(inp1, syn1, out1)
        o_syn2, o_inp2, o_out2 = self.prune(inp2, syn2, out2)

        o_syn1 = sparse.csr_matrix(o_syn1).tolil()
        o_delay1 = self.get_delay(sparse.csr_matrix(o_syn1).tolil(), self.distance_mat)
        o_syn2 = sparse.csr_matrix(o_syn2).tolil()
        o_delay2 = self.get_delay(sparse.csr_matrix(o_syn2).tolil(), self.distance_mat)

        o_out1 = sparse.csr_matrix(o_out1).tolil()
        o_out2 = sparse.csr_matrix(o_out2).tolil()
        o_out_delay1 = self.get_delay(o_out1, self.out_distance_mat)
        o_out_delay2 = self.get_delay(o_out2, self.out_distance_mat)

        o_inp1 = sparse.csr_matrix(o_inp1).tolil()
        o_inp2 = sparse.csr_matrix(o_inp2).tolil()
        o_inp_delay1 = self.get_delay(o_inp1, self.inp_distance_mat)
        o_inp_delay2 = self.get_delay(o_inp2, self.inp_distance_mat)

        o1 = Gene(
            o_inhb1,
            o_vthr1,
            o_vres1,
            o_vleak1,
            o_tsi1,
            o_tmi1,
            o_syn1,
            o_delay1,
            o_inp1,
            o_inp_delay1,
            o_inp_inhb1,
            o_out1,
            o_out_delay1,
        )
        o2 = Gene(
            o_inhb2,
            o_vthr2,
            o_vres2,
            o_vleak2,
            o_tsi2,
            o_tmi2,
            o_syn2,
            o_delay2,
            o_inp2,
            o_inp_delay2,
            o_inp_inhb2,
            o_out2,
            o_out_delay2,
        )
        return o1, o2

    def _del_edge(self, arr):
        del_edge_idx = (np.random.random(arr[arr != 0].shape) > del_edge).astype(int)
        arr[arr != 0] = del_edge_idx * arr[arr != 0]
        return arr

    def _chg_edge(self, arr):
        chg_edge_idx = (np.random.random(arr[arr != 0].shape) < chg_edge).astype(int)
        arr[arr != 0] = (1 - chg_edge_idx) * arr[arr != 0] + chg_edge_idx * (
            np.random.rand(*chg_edge_idx.shape) * (param_range[1] - eps) + eps
        )
        return arr

    def _add_edge(self, arr):
        add_edge_idx = (np.random.random(arr[arr == 0].shape) < add_edge).astype(int)

        arr[arr == 0] = add_edge_idx * (
            np.random.rand(*add_edge_idx.shape) * (param_range[1] - eps) + eps
        )
        return arr

    def _chg_inhb(self, arr):
        chg_inhb_idx = (np.random.random(arr.shape) < chg_inhb).astype(int)
        arr = arr * (1 - chg_inhb_idx) + arr * -1 * chg_inhb_idx
        return arr

    def _chg_vthr(self, arr):
        chg_vthr_idx = (np.random.random(arr.shape) < chg_vthr).astype(int)
        arr = arr * (1 - chg_vthr_idx) + chg_vthr_idx * (
            np.random.rand(*chg_vthr_idx.shape) * (self.vthr_range[1] - eps) + eps
        )
        return arr

    def _chg_vres(self, arr):
        chg_vres_idx = (np.random.random(arr.shape) < chg_vthr).astype(int)
        arr = arr * (1 - chg_vres_idx) + chg_vres_idx * (
            np.random.rand(*chg_vres_idx.shape) * (v_reset_range[1] - v_reset_range[0])
            + v_reset_range[0]
        )
        return arr

    def _chg_vleak(self, arr):
        chg_vleak_idx = (np.random.random(arr.shape) < chg_vthr).astype(int)
        arr = arr * (1 - chg_vleak_idx) + chg_vleak_idx * (
            np.random.rand(*chg_vleak_idx.shape) * (v_leak_range[1] - v_leak_range[0])
            + v_leak_range[0]
        )
        return arr

    def _chg_tau_syn(self, arr):
        chg_tau_syn_idx = (np.random.random(arr.shape) < chg_vthr).astype(int)
        arr = arr * (1 - chg_tau_syn_idx) + chg_tau_syn_idx * (
            np.random.rand(*chg_tau_syn_idx.shape)
            * (tau_syn_range[1] - tau_syn_range[0])
            + tau_syn_range[0]
        )
        return arr

    def _chg_tau_mem(self, arr):
        chg_tau_mem_idx = (np.random.random(arr.shape) < chg_vthr).astype(int)
        arr = arr * (1 - chg_tau_mem_idx) + chg_tau_mem_idx * (
            np.random.rand(*chg_tau_mem_idx.shape)
            * (tau_mem_range[1] - tau_mem_range[0])
            + tau_mem_range[0]
        )
        return arr

    def mutation(self, g):
        """
        Applies mutation to a gene.

        This method introduces random changes to the gene's parameters,
        including adding, deleting, or changing synaptic weights, and modifying
        neuron properties.

        Args:
            g (Gene): The gene to be mutated.

        Returns:
            Gene: The mutated gene.
        """
        # Synapses
        syn = g.synapses.toarray().flatten()
        syn = self._del_edge(syn)
        syn = self._chg_edge(syn)
        syn = self._add_edge(syn)
        syn = syn.reshape(g.synapses.toarray().shape)

        # Neurons
        inhibitions = self._chg_inhb(g.inhibitions)
        thresholds = self._chg_vthr(g.thresholds)
        v_reset = self._chg_vres(g.v_reset)
        v_leak = self._chg_vleak(g.v_leak)
        tau_syn = self._chg_tau_syn(g.tau_syn_inv)
        tau_mem = self._chg_tau_mem(g.tau_mem_inv)

        # Input Synapses
        inp = g.inputs.toarray().flatten()
        inp = self._del_edge(inp)
        inp = self._chg_edge(inp)
        inp = self._add_edge(inp)
        inp = inp.reshape(g.inputs.toarray().shape)

        # Input Neurons
        inp_inhb = self._chg_inhb(g.input_inhibitions)

        # Output Synapses
        out = g.outputs.toarray().flatten()
        out = self._del_edge(out)
        out = self._chg_edge(out)
        out = self._add_edge(out)
        out = out.reshape(g.outputs.toarray().shape)

        # prune the networks to remove isolated loops, unconnected neurons, etc.
        syn, inp, out = self.prune(inp, syn, out)

        syn = sparse.csr_matrix(syn).tolil()
        delay = self.get_delay(syn, self.distance_mat)
        inp = sparse.csr_matrix(inp).tolil()
        inp_delay = self.get_delay(inp, self.inp_distance_mat)
        out = sparse.csr_matrix(out).tolil()
        out_delay = self.get_delay(out, self.out_distance_mat)

        return Gene(
            inhibitions,
            thresholds,
            v_reset,
            v_leak,
            tau_syn,
            tau_mem,
            syn,
            delay,
            inp,
            inp_delay,
            inp_inhb,
            out,
            out_delay,
        )
