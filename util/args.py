import argparse


def str2bool(v):
    """
    Convert string representations to boolean values.

    Args:
        v: Input value (string or boolean)

    Returns:
        bool: Converted boolean value

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class dotdict(dict):
    """
    Dictionary that supports dot notation access to attributes.

    Example:
        d = dotdict({'key': 'value'})
        print(d.key)  # Same as d['key']
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore


class Parser:
    """
    Command-line argument parser for the SNN evolution framework.

    This class defines all available command-line arguments for configuring
    the evolutionary algorithm, network architecture, training parameters,
    and environment settings.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Evolve Spatially Embedded Recurrent Spiking Neural Networks for Control Tasks"
        )

        # =================================================================
        # GENERAL CONFIGURATION
        # =================================================================
        self.parser.add_argument(
            "-c",
            "--checkpoint_path",
            type=str,
            default=None,
            help="Path to directory containing saved model checkpoint for loading pretrained weights",
        )

        self.parser.add_argument(
            "--test",
            type=str2bool,
            nargs="?",
            const=False,
            default=False,
            help="Enable test mode: skip training and only run inference/visualization with existing checkpoint",
        )

        self.parser.add_argument(
            "-r",
            "--random_seed",
            type=int,
            default=0,
            help="Random seed for reproducible experiments (affects network initialization and environment)",
        )

        self.parser.add_argument(
            "--device",
            type=str,
            default="0",
            help="Computing device: GPU ID (e.g., '0', '1') or 'cpu' for CPU-only execution",
        )

        # =================================================================
        # ENVIRONMENT CONFIGURATION
        # =================================================================
        self.parser.add_argument(
            "-g",
            "--game_name",
            type=str,
            default="CartPole-v1",
            help="Gymnasium environment name (e.g., 'CartPole-v1', 'Hopper-v4', 'Ant-v4')",
        )

        self.parser.add_argument(
            "-v",
            "--visualization",
            type=str2bool,
            nargs="?",
            const=False,
            default=True,
            help="Enable real-time visualization of agent performance during training/testing",
        )

        self.parser.add_argument(
            "--max_env_steps",
            type=int,
            default=10,
            help="Maximum number of environment steps per episode before automatic reset",
        )

        self.parser.add_argument(
            "-d",
            "--discretize_intervals",
            type=int,
            default=0,
            help="For continuous action spaces: number of discrete intervals per action dimension. "
            "Set to 0 or 1 to keep actions continuous",
        )

        # =================================================================
        # NETWORK ARCHITECTURE
        # =================================================================
        self.parser.add_argument(
            "-n",
            "--net_size",
            nargs="+",
            type=int,
            default=[8, 8, 1],
            help="3D grid dimensions for hidden layer neurons: [width, height, depth]. "
            "Total hidden neurons = width × height × depth",
        )

        self.parser.add_argument(
            "--spike_steps",
            type=int,
            default=64,
            help="Number of SNN simulation timesteps per environment step (higher = more temporal dynamics)",
        )

        self.parser.add_argument(
            "--max_vthr",
            type=int,
            default=1000,
            help="Maximum membrane potential threshold for LIF neurons (controls spiking sensitivity)",
        )

        self.parser.add_argument(
            "--spatial",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Enable spatial embedding: connection probabilities decay with distance in 3D space",
        )

        self.parser.add_argument(
            "--prune_unconnected",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Remove isolated neurons and disconnected subgraphs to improve efficiency",
        )

        # =================================================================
        # EVOLUTIONARY ALGORITHM PARAMETERS
        # =================================================================
        self.parser.add_argument(
            "-i",
            "--num_iterations",
            type=int,
            default=1000,
            help="Total number of evolutionary generations to run",
        )

        self.parser.add_argument(
            "--num_gene_samples",
            type=int,
            default=100,
            help="Population size: number of individual networks evaluated per generation",
        )

        self.parser.add_argument(
            "--evolution_method",
            type=str,
            default="classic",
            choices=["classic", "map_elites"],
            help="Evolutionary strategy: 'classic' (standard GA) or 'map_elites' (quality-diversity)",
        )

        # =================================================================
        # MAP-ELITES SPECIFIC PARAMETERS
        # =================================================================
        self.parser.add_argument(
            "--sigma_bins",
            type=int,
            default=10,
            help="MAP-Elites: Number of bins for first behavioral descriptor (connection strength diversity)",
        )

        self.parser.add_argument(
            "--sparsity_bins",
            type=int,
            default=10,
            help="MAP-Elites: Number of bins for second behavioral descriptor (network sparsity)",
        )

        # =================================================================
        # TRAINING AND EVALUATION
        # =================================================================
        self.parser.add_argument(
            "--batch_size_gene",
            type=int,
            default=100,
            help="Number of network configurations to evaluate in parallel on GPU (memory permitting)",
        )

        self.parser.add_argument(
            "--num_data_samples",
            type=int,
            default=3,
            help="Total number of evaluation episodes per network (higher = more reliable fitness estimate)",
        )

        self.parser.add_argument(
            "-b",
            "--batch_size_data",
            type=int,
            default=3,
            help="Number of parallel environments to run simultaneously per network evaluation",
        )

        self.parser.add_argument(
            "--curiculum_learning",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Gradually increase episode length by 1.1x every generation for progressive difficulty. Capped at 1000 steps.",
        )

    def parse_args(self):
        """
        Parse command-line arguments and return configuration object.

        Returns:
            argparse.Namespace: Parsed arguments accessible via dot notation
        """
        args = self.parser.parse_args()
        return args
