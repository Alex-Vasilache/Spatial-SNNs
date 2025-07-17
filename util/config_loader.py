import yaml
import argparse
import os
from .args import dotdict


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


def map_config_to_args(config_dict):
    """
    Map the nested YAML configuration to the original argument names.

    Args:
        config_dict (dict): The nested configuration dictionary

    Returns:
        dict: Flattened dictionary with original argument names
    """
    # Mapping from YAML structure to original argument names
    mapping = {
        # General configuration
        "general_checkpoint_path": "checkpoint_path",
        "general_test": "test",
        "general_random_seed": "random_seed",
        "general_device": "device",
        # Environment configuration
        "environment_game_name": "game_name",
        "environment_visualization": "visualization",
        "environment_max_env_steps": "max_env_steps",
        "environment_discretize_intervals": "discretize_intervals",
        # Network architecture
        "network_net_size": "net_size",
        "network_spike_steps": "spike_steps",
        "network_max_vthr": "max_vthr",
        "network_spatial": "spatial",
        "network_prune_unconnected": "prune_unconnected",
        # Evolutionary algorithm parameters
        "evolution_num_iterations": "num_iterations",
        "evolution_num_gene_samples": "num_gene_samples",
        "evolution_evolution_method": "evolution_method",
        # MAP-Elites specific parameters
        "map_elites_sigma_bins": "sigma_bins",
        "map_elites_sparsity_bins": "sparsity_bins",
        # Training and evaluation
        "training_batch_size_gene": "batch_size_gene",
        "training_num_data_samples": "num_data_samples",
        "training_batch_size_data": "batch_size_data",
        "training_curiculum_learning": "curiculum_learning",
    }

    # Flatten the nested structure
    flattened = flatten_config(config_dict)

    # Map to original argument names
    mapped_config = {}
    for key, value in flattened.items():
        if key in mapping:
            mapped_config[mapping[key]] = value
        else:
            # Keep original key if no mapping found
            mapped_config[key] = value

    return mapped_config


def flatten_config(config_dict, parent_key="", sep="_"):
    """
    Flatten a nested dictionary by concatenating nested keys with separator.

    Args:
        config_dict (dict): The nested dictionary to flatten
        parent_key (str): The parent key for recursion
        sep (str): Separator to use between nested keys

    Returns:
        dict: Flattened dictionary
    """
    items = []
    for k, v in config_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config_from_yaml(config_path):
    """
    Load configuration from YAML file and convert to dotdict format.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dotdict: Configuration object with flattened structure
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Map the nested structure to original argument names
    mapped_config = map_config_to_args(config)

    # Convert to dotdict
    return dotdict(mapped_config)


def create_parser():
    """
    Create argument parser for command-line overrides.

    Returns:
        argparse.ArgumentParser: Parser for command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Evolve Spatially Embedded Recurrent Spiking Neural Networks for Control Tasks"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )

    # Add command-line override options for key parameters
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Override checkpoint path from config",
    )

    parser.add_argument(
        "--test",
        type=str2bool,
        nargs="?",
        const=False,
        default=None,
        help="Override test mode from config",
    )

    parser.add_argument(
        "--random_seed", type=int, default=None, help="Override random seed from config"
    )

    parser.add_argument(
        "--device", type=str, default=None, help="Override device from config"
    )

    parser.add_argument(
        "--game_name", type=str, default=None, help="Override game name from config"
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=None,
        help="Override number of iterations from config",
    )

    return parser


def load_config(config_path=None, override_args=None):
    """
    Load configuration from YAML file with optional command-line overrides.

    Args:
        config_path (str, optional): Path to YAML config file. If None, uses default.
        override_args (dict, optional): Dictionary of arguments to override from config

    Returns:
        dotdict: Configuration object
    """
    # Load base configuration from YAML
    if config_path is None:
        config_path = "config.yaml"

    config = load_config_from_yaml(config_path)

    # Apply command-line overrides if provided
    if override_args:
        for key, value in override_args.items():
            if value is not None:  # Only override if value is not None
                config[key] = value

    return config
