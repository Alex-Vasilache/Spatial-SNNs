from .models.norse_dynamic import Net
import gymnasium as gym
import os
import time
import numpy as np
import json
from src.search.evolution import Gene, merge_genes
from src.encoder import SimpleEncoder as Encoder
from util.scaler import create_scaler
from gymnasium.wrappers import TimeLimit
from gymnasium.spaces import Box

from tqdm import tqdm

NUM_FEATURES = 2

MAX_STD = 0.5


class Task:
    """
    Manages the evaluation of neural networks in a given gym environment.

    This class handles the setup of the environment, the network, and the encoder.
    It orchestrates the evaluation of a pool of genes, computes their rewards,
    and calculates feature descriptors based on their performance.
    """

    def __init__(self, args):
        """
        Initializes the Task object.

        Args:
            args (dotdict): A dictionary-like object containing configuration parameters.
        """
        self.batch_size_data = args.batch_size_data
        self.batch_size_gene = args.batch_size_gene
        self.batch_size = self.batch_size_data * self.batch_size_gene
        self.num_data_samples = args.num_data_samples
        self.num_gene_samples = args.num_gene_samples
        self.num_data_batches = int(self.num_data_samples / self.batch_size_data)
        self.num_gene_batches = int(self.num_gene_samples / self.batch_size_gene)
        if (
            args.max_env_steps == -1
            or args.max_env_steps is None
            or args.max_env_steps == 0
        ):
            self.envs = np.array(
                [gym.make(args.game_name) for i in range(self.batch_size)]
            )
        else:
            self.envs = np.array(
                [
                    gym.make(args.game_name, max_episode_steps=int(args.max_env_steps))
                    for i in range(self.batch_size)
                ]
            )

        self.game_name = args.game_name

        self.spike_steps = args.spike_steps

        if not hasattr(args, "data_path") or args.data_path is None:
            self.data_path = "output/"
            self.data_path = os.path.join(
                self.data_path,
                args.game_name + "-" + args.time,
            )
        else:
            self.data_path = args.data_path

        if not args.test:
            if not (os.path.exists(self.data_path)):
                os.makedirs(self.data_path)

        if args.discretize_intervals in [None, 0, 1]:
            args.discretize_intervals = None

        if isinstance(self.envs[0].action_space, Box):
            self.continuous_action_space = True
            self.out_feat = self.envs[0].action_space.shape[0]
            self.action_bounds = np.stack(
                [self.envs[0].action_space.low, self.envs[0].action_space.high]
            ).T

        else:
            self.continuous_action_space = False
            self.out_feat = self.envs[0].action_space.n
            args.discretize_intervals = None
            self.action_bounds = None

        if self.continuous_action_space:
            self.of = self.out_feat
        else:
            self.of = 1

        self.discretize_intervals = args.discretize_intervals
        self.viz = args.visualization
        self.size = args.net_size
        self.inp_feat = self.envs[0].observation_space.shape[0]

        scaler = create_scaler(env=self.envs[0])

        self.encoder = Encoder(
            self.inp_feat,
            self.batch_size,
            scaler
        )

        self.hid_feat = args.net_size

        if not args.test:
            with open(os.path.join(self.data_path, "args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=2)

        self.args = args
        self.net = None

    def set_params(self, params):
        """
        Sets the parameters of the neural network.

        This method merges a list of genes and initializes or updates the
        neural network with the new parameters.

        Args:
            params (list): A list of Gene objects.
        """
        merged_gene = merge_genes(params)
        if self.net is None or self.net.batch_size_gene != self.batch_size_gene:
            self.net = Net(
                size=self.size,
                batch_size_gene=self.batch_size_gene,
                batch_size_data=self.batch_size_data,
                init_gene=merged_gene,
                action_bounds=self.action_bounds,
                num_steps=self.args.spike_steps,
                input_features=self.inp_feat,
                output_features=self.out_feat,
                discretize_intervals=self.discretize_intervals
                if self.discretize_intervals is not None
                else 0,
                device=(
                    "cuda:" + str(self.args.device)
                    if self.args.device is not None
                    else "cpu"
                ),
            )
        else:
            self.net.set_gene(merged_gene)

    def reward(self, pool):
        """
        Evaluates a pool of genes and returns their performance metrics.

        This method iterates through batches of genes and data samples,
        evaluates their rollouts in the environment, and computes their
        rewards and feature descriptors.

        Args:
            pool (list): A list of Gene objects to be evaluated.

        Returns:
            tuple: A tuple containing the mean rewards, standard deviation of
                   rewards, and a stack of feature descriptors.
        """
        gene_batches = np.split(
            np.array(pool, dtype=object), self.num_gene_batches
        )
        data = np.arange(self.num_data_samples)
        data_batches = np.split(
            np.repeat(data, self.batch_size_gene).reshape(
                self.num_data_samples, self.batch_size_gene
            ),
            self.num_data_batches,
        )

        rewards = np.zeros((self.num_gene_samples, self.num_data_samples))
        ft_desc = np.zeros(
            (self.num_gene_samples, self.num_data_samples, self.of, NUM_FEATURES)
        )

        for i, genes in enumerate(gene_batches):
            self.set_params(list(genes))
            for j, datas in enumerate(data_batches):
                if self.net:
                    self.net.stop()
                batch_reward, batch_ft_desc = self.evaluate_rollout(datas)

                batch_reward = batch_reward.reshape(
                    self.batch_size_data, self.batch_size_gene
                ).T

                rewards[
                    i * self.batch_size_gene : (i + 1) * self.batch_size_gene,
                    j * self.batch_size_data : (j + 1) * self.batch_size_data,
                ] = batch_reward

                ft_desc[
                    i * self.batch_size_gene : (i + 1) * self.batch_size_gene,
                    j * self.batch_size_data : (j + 1) * self.batch_size_data,
                ] = batch_ft_desc

        # Number of used joints
        f1 = np.sum(ft_desc.mean(1)[:, :, 0] != 0, axis=1)

        f2 = np.sum(ft_desc.mean(1)[:, :, 0] * (ft_desc.mean(1)[:, :, 0] != 0), axis=1)
        f2[f1 != 0] /= f1[f1 != 0]
        # Make values of f2 that are close to 0 (within 0.01) actually 0
        f2[np.abs(f2) < 0.01] = 0

        sparsity = self.compute_sparsity(np.array(gene_batches).flatten())

        total_steps = ft_desc[:, :, 0, -1].sum(1)

        return (
            rewards.mean(axis=1),
            rewards.std(axis=1),
            np.stack([f1, f2, sparsity, total_steps]),
        )

    def evaluate_rollout(self, datas):
        """
        Performs a rollout in the environment for a batch of networks.

        This method resets the environment, then steps through the environment
        for a maximum number of steps, collecting rewards and network outputs.

        Args:
            datas (np.ndarray): An array of data samples to use for seeding the environment.

        Returns:
            tuple: A tuple containing the total scores and feature descriptors for the rollout.
        """
        states = self._reset(datas)
        scores = np.zeros(self.batch_size)
        outputs = []
        dones = [False] * self.batch_size
        truncs = [False] * self.batch_size

        for _ in tqdm(
            range(int(self.args.max_env_steps)),
            desc="Evaluating rollout",
            leave=False,
        ):
            if all(dones) or all(truncs):
                break
            step_outputs = self._step(states, (dones or truncs))
            states, rewards, dones, truncs, sigmoid_actions = (
                step_outputs[0],
                step_outputs[1],
                step_outputs[2],
                step_outputs[3],
                step_outputs[4]
            )

            scores += rewards
            if isinstance(sigmoid_actions, np.ndarray):
                sigmoid_actions[np.array(dones)] = None
            outputs.append(sigmoid_actions)

        if self.net:
            self.net.stop()
        # del self.net
        outputs = np.stack(outputs).reshape(
            len(outputs), self.batch_size_data, self.batch_size_gene, self.of
        )

        ft_desc = self.compute_feature_descriptors(outputs)
        return (scores, ft_desc)

    def compute_sparsity(self, genes):
        """
        Computes the sparsity of the connections in a list of genes.

        Args:
            genes (list): A list of Gene objects.

        Returns:
            np.ndarray: An array containing the sparsity value for each gene.
        """
        sparsity = np.zeros(len(genes))
        for i, gene in enumerate(genes):
            tot = (
                np.prod(gene.inputs.shape)
                + np.prod(gene.outputs.shape)
                + np.prod(gene.synapses.shape)
            )
            nnz = gene.inputs.nnz + gene.outputs.nnz + gene.synapses.nnz
            sparsity[i] = nnz / tot

        return sparsity

    def compute_feature_descriptors(self, outputs):
        """
        Computes feature descriptors from the network's output signals.

        The feature descriptors include the standard deviation of the signal
        and the number of steps the network was active.

        Args:
            outputs (np.ndarray): The output signals from the network.

        Returns:
            np.ndarray: An array of feature descriptors.
        """
        # outputs: (ts, bs_data, bs_gene, out_feat)
        ts = outputs.shape[0]
        outputs = outputs.transpose(2, 1, 3, 0)
        # outputs: (bs_gene, bs_data, out_feat, ts)
        outputs = outputs.reshape(-1, ts)
        # count non-nan values
        steps = np.sum(~np.isnan(outputs), axis=1)
        ft_desc = np.zeros((outputs.shape[0], NUM_FEATURES))

        for i in range(outputs.shape[0]):
            # mask nan values
            nan_mask = ~np.isnan(outputs[i])
            signal = outputs[i][nan_mask]
            f1 = np.std(signal) / (MAX_STD + np.finfo(float).eps)
            if not (0 <= f1 < 1):
                f1 = 0
            ft_desc[i] = [f1, steps[i]]

        ft_desc = ft_desc.reshape(
            self.batch_size_gene, self.batch_size_data, self.of, NUM_FEATURES
        )
        return ft_desc

    def _reset(self, seeds):
        """
        Resets the environments with the given seeds.

        Args:
            seeds (np.ndarray): An array of seeds for resetting the environments.

        Returns:
            np.ndarray: The initial states of the environments after reset.
        """
        resets = [
            env.reset(seed=int(seed * 1000 + self.args.random_seed))
            for env, seed in zip(self.envs, seeds.flatten())
        ]
        states = np.array([r[0] for r in resets])
        self.encoder.reset(states)
        return states

    def _step(self, states, dones):
        """
        Performs a single step in the environment for all parallel agents.

        This method normalizes the states, passes them through the network,
        and executes the resulting actions in the environment.

        Args:
            states (np.ndarray): The current states of the environments.
            dones (list): A list of booleans indicating if each environment is done.

        Returns:
            tuple: A tuple containing the next states, rewards, done flags,
                   truncation flags, and the sigmoid actions from the network.
        """
        norm_states = self.encoder.forward(states)

        norm_states = (norm_states - 0.5) * 5
        t, b, f = self.spike_steps, self.batch_size, self.inp_feat
        input_state = (
            norm_states.reshape(b, f)
            .repeat(t)
            .reshape(b, f, t)
            .swapaxes(0, 2)
            .swapaxes(1, 2)
        )

        done_mask = (
            (1 - np.array(dones))
            .repeat(self.spike_steps)
            .reshape(self.batch_size, self.spike_steps)
            .swapaxes(0, 1)
            .repeat(self.inp_feat)
            .reshape(self.spike_steps, self.batch_size, self.inp_feat)
        )
        input_state = input_state * done_mask
        input_state = input_state.reshape(
            self.spike_steps, self.batch_size_data, self.batch_size_gene * self.inp_feat
        )

        actions = None
        sigmoid_actions = None

        if self.net:
            actions, sigmoid_actions = self.net.forward(input_state)

        if actions is None or sigmoid_actions is None:
            raise ValueError("Net not initialized or forward pass failed")

        steps = [
            (
                (
                    env.step(action)
                    if self.continuous_action_space
                    else env.step(action[0])
                )
                if not done
                else (
                    states[i],
                    0.0,
                    True,
                    True,
                    {},
                )
            )
            for (i, env, action, done) in zip(
                range(self.batch_size),
                self.envs,
                actions.reshape(self.batch_size, -1),
                dones,
            )
        ]
        states = np.array([s[0] for s in steps])
        rewards = np.array([s[1] for s in steps])
        dones = [s[2] for s in steps]
        truncs = [s[3] for s in steps]
        return (
            states,
            rewards,
            dones,
            truncs,
            sigmoid_actions.reshape(self.batch_size, -1),
        )

    def cleanup(self):
        """
        Cleans up resources, specifically the network object.
        """
        if "net" in self.__dict__.keys():
            del self.net
