#!/usr/bin/env python3
"""
Test script to verify YAML configuration loading works correctly.
This script tests the configuration system without importing the full Spatial-SNNs modules.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.config_loader import load_config, create_parser
from util.args import dotdict


def test_config_loading():
    """Test that configuration loading works correctly."""
    print("Testing YAML configuration loading...")

    # Test basic configuration loading
    config = load_config()

    # Check that all expected parameters are present
    expected_params = [
        "checkpoint_path",
        "test",
        "random_seed",
        "device",
        "game_name",
        "visualization",
        "max_env_steps",
        "discretize_intervals",
        "net_size",
        "spike_steps",
        "max_vthr",
        "spatial",
        "prune_unconnected",
        "num_iterations",
        "num_gene_samples",
        "evolution_method",
        "sigma_bins",
        "sparsity_bins",
        "batch_size_gene",
        "num_data_samples",
        "batch_size_data",
        "curiculum_learning",
    ]

    missing_params = []
    for param in expected_params:
        if not hasattr(config, param):
            missing_params.append(param)

    if missing_params:
        print(f"❌ Missing parameters: {missing_params}")
        return False

    print("✅ All expected parameters present")

    # Test some specific values
    print(f"Game name: {config.game_name}")
    print(f"Random seed: {config.random_seed}")
    print(f"Network size: {config.net_size}")
    print(f"Number of iterations: {config.num_iterations}")
    print(f"Device: {config.device}")

    return True


def test_command_line_overrides():
    """Test that command-line overrides work correctly."""
    print("\nTesting command-line overrides...")

    # Simulate command-line arguments
    override_args = {"game_name": "Hopper-v4", "random_seed": 42, "num_iterations": 500}

    config = load_config(override_args=override_args)

    # Check that overrides were applied
    if (
        config.game_name == "Hopper-v4"
        and config.random_seed == 42
        and config.num_iterations == 500
    ):
        print("✅ Command-line overrides working correctly")
        return True
    else:
        print("❌ Command-line overrides not working correctly")
        return False


def test_parser():
    """Test that the argument parser works correctly."""
    print("\nTesting argument parser...")

    parser = create_parser()

    # Check that parser has expected arguments
    expected_args = [
        "config",
        "checkpoint_path",
        "test",
        "random_seed",
        "device",
        "game_name",
        "num_iterations",
    ]

    parser_args = [action.dest for action in parser._actions]

    missing_args = []
    for arg in expected_args:
        if arg not in parser_args:
            missing_args.append(arg)

    if missing_args:
        print(f"❌ Missing parser arguments: {missing_args}")
        return False

    print("✅ Argument parser has all expected arguments")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing YAML Configuration System")
    print("=" * 50)

    tests = [test_config_loading, test_command_line_overrides, test_parser]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)

    if passed == total:
        print("🎉 All tests passed! YAML configuration system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration system.")
        return 1


if __name__ == "__main__":
    exit(main())
