import numpy as np
from util.args import Parser
from src.train import train
from src.visualizer import Visualizer
from threading import Thread
import os
import time
import warnings
from util.data import load_checkpoint
from util.args import dotdict
from scipy.sparse._base import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = Parser()
args = parser.parse_args()
np.random.seed(args.random_seed)


def main(args):
    """
    Main function to run the training or visualization process.

    This function serves as the entry point for the application. It handles
    command-line arguments to determine whether to run the training process
    or to visualize a pre-trained agent.

    Args:
        args (dotdict): A dictionary-like object containing configuration parameters.
    """

    if not args.test:
        # --- Training Mode ---
        data_path = "output/"
        args.time = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.checkpoint_path:
            search_dist, tests, new_args, _, _ = load_checkpoint(args.checkpoint_path)
            args.game_name = dotdict(new_args).game_name

        if args.game_name is None:
            raise ValueError("Game name cannot be None")
        data_path = os.path.join(data_path, args.game_name + "-" + args.time)

        args.data_path = data_path

        if args.visualization:
            visualizer = Visualizer(data_path)
            viz_thread = Thread(target=visualizer.start)
            viz_thread.start()

        train(args)

        if args.visualization:
            viz_thread.join()

    else:
        # --- Visualization Mode ---
        visualizer = Visualizer(args.checkpoint_path)
        visualizer.start()


if __name__ == "__main__":
    main(args=args)
