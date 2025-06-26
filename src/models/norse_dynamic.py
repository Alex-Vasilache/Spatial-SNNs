import numpy as np
from src.search.evolution import Gene

import torch
from norse.torch import LIFCell, LIFParameters
from norse.torch.functional.lif import LIFFeedForwardState
import warnings
from scipy.sparse._base import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Net:
    """
    A dynamic spiking neural network model implemented using Norse.

    This class encapsulates the network structure, including neurons and synapses,
    and manages its state and evolution over time. It is designed to be
    flexible for use in reinforcement learning tasks, supporting both
    discrete and continuous action spaces.

    Attributes:
        input_features (int): The number of input features.
        output_features (int): The number of output features.
        num_steps (int): The number of simulation steps to run per forward pass.
        continuous_action_space (bool): Flag indicating if the action space is continuous.
        batch_size_gene (int): The batch size for genes.
        batch_size_data (int): The batch size for data.
        device (str): The device to run the computations on (e.g., 'cuda' or 'cpu').
        gene (Gene): The gene object containing the network's parameters.
        size (list): The dimensions of the network.
        n (int): The total number of neurons in the network.
    """

    def __init__(
        self,
        size,
        batch_size_gene: int,
        batch_size_data: int,
        init_gene: Gene,
        action_bounds=None,
        num_steps=1,
        input_features=8,
        output_features=1,
        discretize_intervals=3,
        device="cuda",
    ):
        # np.random.seed(42)
        self.input_features = input_features
        self.output_features = output_features
        self.num_steps = num_steps
        self.continuous_action_space = False
        self.batch_size_gene = batch_size_gene
        self.batch_size_data = batch_size_data
        self.device = device
        self.gene = init_gene

        self.size = size
        self.n = np.prod(size)

        if not action_bounds is None:
            self.continuous_action_space = True
            self.discretize_intervals = discretize_intervals
            self.action_bounds = action_bounds
            self.action_range = action_bounds[:, 1] - action_bounds[:, 0]
            if discretize_intervals:
                self.action_steps = self.action_range / (self.discretize_intervals - 1)
                self.output_features = int(output_features * discretize_intervals)

        # Define Neurons
        self.neurons = LIFCell().to(self.device)

        # Define Synapses

        self.synapses = torch.nn.Linear(
            in_features=(self.batch_size_gene * self.n),
            out_features=(self.batch_size_gene * self.n),
            bias=False,
            device=self.device,
        )
        self.input_synapses = torch.nn.Linear(
            in_features=(self.batch_size_gene * self.input_features),
            out_features=(self.batch_size_gene * self.n),
            bias=False,
            device=self.device,
        )
        self.output_synapses = torch.nn.Linear(
            in_features=(self.batch_size_gene * self.n),
            out_features=(self.batch_size_gene * self.output_features),
            bias=False,
            device=self.device,
        )

        self.reset()

    def forward(self, state):
        with torch.no_grad():
            self.set_input(state)
            self.run()

            output_spikes = self.get_output()

            if not self.continuous_action_space:
                action = np.argmax(output_spikes, axis=2)
                # action = self.output_features - 1 - action
                sigmoid_outputs = action / (self.output_features - 1)
            elif self.discretize_intervals:
                output_spikes = output_spikes.reshape(
                    self.batch_size_data,
                    self.batch_size_gene,
                    -1,
                    self.discretize_intervals,
                )
                action_intervals = np.argmax(output_spikes, axis=-1)
                action = self.action_bounds[:, 0] + action_intervals * self.action_steps
                sigmoid_outputs = action_intervals / (self.discretize_intervals - 1)
                # action = action * -1
            else:
                rescale_val = 32
                output_spikes = output_spikes / rescale_val

                # sigmoid
                sigmoid_outputs = 1 / (1 + np.exp(-output_spikes))

                # action range
                action = sigmoid_outputs * self.action_range

                # action bounds
                action = action + self.action_bounds[:, 0]

                action = np.clip(
                    output_spikes,
                    self.action_bounds[:, 0],
                    self.action_bounds[:, 1],
                )

            return action, sigmoid_outputs

    def set_input(self, s_in):
        self.spike_input = torch.tensor(
            s_in, dtype=torch.float32, requires_grad=False
        ).to(self.device)
        # set spike potentials (multiply with linear)

    def get_output(self):
        # (num_steps, batch * out_features)
        output = self.spike_output.detach().cpu().numpy()
        output = output.reshape(
            self.num_steps,
            self.batch_size_data,
            self.batch_size_gene,
            self.output_features,
        )
        output = output.mean(axis=0)

        return output

    def run(self):
        with torch.no_grad():
            for n in range(self.num_steps):
                rec_potentials = self.synapses(self.spike_hidden)
                inp_potentials = self.input_synapses(self.spike_input[n])
                self.spike_hidden, self.s = self.neurons(
                    inp_potentials + rec_potentials, self.s
                )
                self.spike_output[n] = self.output_synapses(self.spike_hidden)

    def stop(self):
        self._reset_accum()

    def pause(self):
        pass

    def _reset_accum(self):
        self.spike_output = torch.tensor(
            np.zeros(
                (
                    self.num_steps,
                    self.batch_size_data,
                    self.batch_size_gene * self.output_features,
                )
            ),
            requires_grad=False,
            dtype=torch.float32,
        ).to(self.device)
        self.spike_input = torch.tensor(
            np.zeros(
                (
                    self.num_steps,
                    self.batch_size_data,
                    self.batch_size_gene * self.input_features,
                )
            ),
            requires_grad=False,
            dtype=torch.float32,
        ).to(self.device)
        self.spike_hidden = torch.tensor(
            np.zeros(
                (
                    self.batch_size_data,
                    self.batch_size_gene * self.n,
                )
            ),
            requires_grad=False,
            dtype=torch.float32,
        ).to(self.device)

    def reset(self):
        # Reset Neurons
        self.lif_params = LIFParameters(
            v_th=torch.tensor(
                self.gene.thresholds, dtype=torch.float32, requires_grad=False
            ).to(self.device),
            v_leak=torch.tensor(
                self.gene.v_leak, dtype=torch.float32, requires_grad=False
            ).to(self.device),
            v_reset=torch.tensor(
                self.gene.v_reset, dtype=torch.float32, requires_grad=False
            ).to(self.device),
            tau_mem_inv=torch.tensor(
                self.gene.tau_mem_inv, dtype=torch.float32, requires_grad=False
            ).to(self.device),
            tau_syn_inv=torch.tensor(
                self.gene.tau_syn_inv, dtype=torch.float32, requires_grad=False
            ).to(self.device),
        )
        self.neurons = LIFCell(self.lif_params).to(self.device)

        # Reset Synapses

        self.synapses.weight.data = torch.tensor(
            self.gene.delays.toarray()
            * self.gene.synapses.toarray()
            * self.gene.inhibitions,
            requires_grad=False,
            dtype=torch.float32,
        ).to(self.device)
        self.input_synapses.weight.data = (
            torch.tensor(
                self.gene.inputs.toarray()
                * self.gene.inp_delays.toarray()
                * self.gene.input_inhibitions,
                requires_grad=False,
                dtype=torch.float32,
            )
            # .to_sparse_csr()
            .to(self.device)
        )
        self.output_synapses.weight.data = (
            torch.tensor(
                self.gene.outputs.toarray()
                * self.gene.out_delays.toarray()
                * self.gene.inhibitions,
                requires_grad=False,
                dtype=torch.float32,
            )
            # .to_sparse_csr()
            .to(self.device)
        )

        v_init = self.neurons.p.v_reset.repeat(self.batch_size_data, 1)
        self.s = LIFFeedForwardState(
            v=v_init,
            i=torch.zeros_like(v_init),
        )
        self._reset_accum()

    def set_gene(self, gene):
        self.gene = gene
        self.reset()
