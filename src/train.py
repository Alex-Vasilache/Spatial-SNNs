import os
from src.search.evolution import Evolution
from util.training import ProgressBar
from util.data import save_weights, load_checkpoint
from src.task import Task
import numpy as np
from scipy.sparse._base import SparseEfficiencyWarning
import warnings
from util.args import dotdict

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def train(args):
    """
    Main training loop for the evolutionary algorithm.

    This function orchestrates the training process, including setting up the
    environment and evolution, running the training iterations, evaluating
    elites, and saving checkpoints.

    Args:
        args (dotdict): A dictionary-like object containing configuration parameters.
    """
    if args.checkpoint_path:
        path = args.checkpoint_path
        device = args.device
        args_time = args.time
        search_dist, tests, new_args, train_log, test_log = load_checkpoint(args.checkpoint_path)
        args.__dict__ = new_args
        weights_iteration = args.weights_iteration
        args.data_path = None
        args.checkpoint_path = path
        args.device = device
        args.time = args_time
        args.weights_iteration = weights_iteration
        tests = [], [], [], []
    else:
        tests = [], [], [], []

    if "max_env_steps" not in args:
        if args.curiculum_learning:
            args.max_env_steps = 10
        else:
            args.max_env_steps = 1000
    else:
        if args.curiculum_learning:
            args.max_env_steps = args.max_env_steps
        else:
            args.max_env_steps = args.max_env_steps
    task = Task(args)
    data_path = task.data_path
    args.data_path = data_path
    increase_every_nth = 1

    output_features = task.out_feat

    if not task.action_bounds is None:
        if task.discretize_intervals:
            output_features = int(output_features * task.discretize_intervals)

    if args.checkpoint_path:
        if args.evolution_method == "map_elites":
            search_dist.update_elites()
            search_dist.update_pool()

        print("Succesfully loaded weights at:\n" + path)
    else:
        search_dist = Evolution(
            net_size=args.net_size,
            pool_size=args.num_gene_samples,
            input_neurons=task.inp_feat,
            output_neurons=output_features,
            f1=task.of,
            f2=args.sigma_bins,
            f3=args.sparsity_bins,
            max_vthr=args.max_vthr,
            prune_unconnected=args.prune_unconnected,
            evolution_method=args.evolution_method,
            spatial=args.spatial
        )

    info = args.__dict__
    info["weights_iteration"] = 0
    info["avg_score"] = "None"
    info["max_score"] = "None"
    info["map_scores"] = np.array2string(
        np.array(search_dist.map_scores),
        formatter={"float_kind": lambda x: "%4.0f" % x},
    )

    map_scores = np.array(search_dist.map_scores)
    # Ensure map_scores is a 2D array for saving
    if len(map_scores.shape) == 1:
        map_scores = map_scores.reshape(-1, 1)
    elif len(map_scores.shape) == 3:
        map_scores = map_scores.reshape(-1, map_scores.shape[-1])

    np.savetxt(
        os.path.join(data_path, task.game_name + ".txt"),
        map_scores,
        fmt="%4.0f",
    )

    test_elites = tests[0]
    test_scores = tests[1]
    test_ftdesc = tests[2]
    test_stds = tests[3]

    train_log = []
    test_log = []

    save_weights(
        search_dist,
        (test_elites, test_scores, test_ftdesc),
        data_path,
        info,
        train_log,
        test_log,
    )

    test_args = dotdict(args.__dict__.copy())
    test_args.test = True
    test_args.random_seed = 99
    test_args.num_data_samples = 100
    test_args.batch_size_data = 100
    test_args.max_env_steps = 1000
    total_steps = 0

    progress_bar = ProgressBar(num_iterations=args.num_iterations)

    for iteration in range(args.weights_iteration, args.num_iterations):
        samples = search_dist.pool
        rewards, std_devs, ft_desc = task.reward(samples)

        search_dist.update_params(rewards, ft_desc[:-1])
        total_steps += int(ft_desc[-1].sum())

        max_rew_idx = np.argmax(rewards)

        train_log.append(
            (
                iteration,
                total_steps,
                rewards.mean(),
                rewards.max(),
                std_devs[max_rew_idx],
            )
        )

        info["weights_iteration"] = str(iteration)
        info["total_steps"] = str(total_steps)
        info["avg_reward"] = (
            str(search_dist.scores.mean()) if search_dist.scores.size > 0 else "None"
        )
        info["max_reward"] = (
            str(search_dist.scores.max()) if search_dist.scores.size > 0 else "None"
        )
        if search_dist.evolution_method == "map_elites":
            info["map_scores"] = np.array2string(
                np.array(search_dist.map_scores),
                max_line_width=1000,
                formatter={"float_kind": lambda x: "%4.0f" % x},
            )

        if (
            iteration % 10 == 0 and iteration != 0
        ) or iteration == args.num_iterations - 1:
            # --- Testing and Logging ---
            train_scores = np.copy(search_dist.scores)
            train_elites = np.copy(search_dist.elites)

            # Sort the scores and elites
            sort_idx = np.argsort(train_scores)[::-1]
            top_m = 1  # max(2, min(int(0.1 * len(train_elites)), len(train_elites)))
            train_scores = train_scores[sort_idx][:top_m]
            train_elites = train_elites[sort_idx][:top_m]

            test_batch_elites = []

            for te in train_elites:
                if te not in test_elites:
                    test_batch_elites.append(te)

            test_args.batch_size_gene = 1
            test_args.num_gene_samples = len(test_batch_elites)
            if test_args.num_gene_samples >= 1:
                test_task = Task(test_args)
                test_batch_scores, test_batch_stds, test_batch_ftdesc = (
                    test_task.reward(test_batch_elites)
                )
                test_task.cleanup()
                del test_task
                test_batch_scores = list(test_batch_scores)
                test_batch_stds = list(test_batch_stds)
                test_batch_ftdesc = list(test_batch_ftdesc.T)
            else:
                test_batch_scores = []
                test_batch_stds = []
                test_batch_ftdesc = []

            # add new elites with score and ftdest to arrays and sort by scores
            test_elites.extend(test_batch_elites)
            test_scores.extend(test_batch_scores)
            test_stds.extend(test_batch_stds)
            test_ftdesc.extend(test_batch_ftdesc)
            sort_idx = np.argsort(test_scores)[::-1]
            test_scores = list(np.array(test_scores)[sort_idx])
            test_stds = list(np.array(test_stds)[sort_idx])
            test_elites = list(np.array(test_elites)[sort_idx])
            test_ftdesc = list(np.array(test_ftdesc)[sort_idx])

            # only keep the top m
            test_scores = test_scores[:top_m]
            test_stds = test_stds[:top_m]
            test_elites = test_elites[:top_m]
            test_ftdesc = test_ftdesc[:top_m]

            info["top_m_train"] = np.array2string(
                np.array(train_scores),
                formatter={"float_kind": lambda x: "%4.0f" % x},
            )
            info["top_m_test"] = np.array2string(
                np.array(test_scores),
                formatter={"float_kind": lambda x: "%4.0f" % x},
            )
            info["top_m_std"] = np.array2string(
                np.array(test_stds),
                formatter={"float_kind": lambda x: "%4.0f" % x},
            )

            test_log.append(
                (
                    iteration,
                    total_steps,
                    np.array(test_scores).mean(),
                    np.array(test_scores).max(),
                    np.array(test_stds)[np.argmax(test_scores)],
                )
            )

            save_weights(
                search_dist,
                (test_elites, test_scores, test_ftdesc),
                data_path,
                info,
                train_log,
                test_log,
            )
            if search_dist.evolution_method == "map_elites":
                map_scores = np.array(search_dist.map_scores)
                # Ensure map_scores is a 2D array for saving
                if len(map_scores.shape) == 1:
                    map_scores = map_scores.reshape(-1, 1)
                elif len(map_scores.shape) == 3:
                    map_scores = map_scores.reshape(-1, map_scores.shape[-1])

                np.savetxt(
                    os.path.join(data_path, task.game_name + ".txt"),
                    map_scores,
                    fmt="%4.0f",
                )
        else:
            save_weights(
                search_dist,
                (test_elites, test_scores, test_ftdesc),
                data_path,
                info,
                train_log,
                test_log,
            )

        # --- Curriculum Learning ---
        if iteration % increase_every_nth == 0 and args.curiculum_learning:
            args.max_env_steps = min(args.max_env_steps * 1.1, 1000)
            del task
            task = Task(args)
            increase_every_nth = increase_every_nth + 1
            info["max_env_steps"] = args.max_env_steps

        progress_bar(
            avg_r1=(search_dist.scores.mean() if search_dist.scores.size > 0 else 0),
            min_r1=(search_dist.scores.min() if search_dist.scores.size > 0 else 0),
            max_r1=(search_dist.scores.max() if search_dist.scores.size > 0 else 0),
        )

    task.cleanup()
    del task
