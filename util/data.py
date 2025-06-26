from scipy import sparse
from functools import reduce
import numpy as np
import joblib
import os
import json
import pandas as pd


def get_rows(lils):
    """
    Get all rows from a list of LIL matrices and adjust their indices.

    Args:
        lils (list): A list of sparse.lil_matrix objects.

    Returns:
        np.ndarray: An array of adjusted row indices.
    """
    return np.array(
        reduce(
            lambda a, b: a + b,
            [
                [list(np.array(row) + mat.shape[1] * i) for row in mat.rows]
                for (i, mat) in enumerate(lils)
            ],
        ),
        dtype=object,
    )


def get_data(lils):
    """
    Get all data from a list of LIL matrices.

    Args:
        lils (list): A list of sparse.lil_matrix objects.

    Returns:
        np.ndarray: An array of data values.
    """
    return np.array(
        reduce(
            lambda a, b: a + b,
            [[data for data in mat.data] for mat in lils],
        ),
        dtype=object,
    )


def merge_lils(lils):
    """
    Merge a list of LIL matrices into a single block-diagonal matrix.

    Args:
        lils (list): A list of sparse.lil_matrix objects.

    Returns:
        sparse.lil_matrix: The merged LIL matrix.
    """
    n_lils = len(lils)
    merged_lil = sparse.lil_matrix(
        (lils[0].shape[0] * n_lils, lils[0].shape[1] * n_lils)
    )
    merged_lil.data = get_data(lils)
    merged_lil.rows = get_rows(lils)
    merged_lil.dtype = lils[0].dtype

    return merged_lil  # type: ignore


def save_weights(search_dist, tests, path, info, train_log, test_log):
    """
    Save the training state, including search distribution, test results,
    and logs.

    Args:
        search_dist: The evolution object.
        tests (tuple): A tuple containing test elites, scores, and feature descriptors.
        path (str): The directory path to save the files.
        info (dict): A dictionary of metadata to save.
        train_log (list): The training log.
        test_log (list): The testing log.
    """
    joblib.dump(search_dist, os.path.join(path, "search_dist.save"))
    joblib.dump(tests, os.path.join(path, "tests.save"))

    with open(os.path.join(path, "args.txt"), "w") as f:
        json.dump(info, f, indent=2)

    train_log = np.array(train_log)
    test_log = np.array(test_log)

    if len(train_log) == 0:
        train_log = train_log.reshape(-1, 5)
    if len(test_log) == 0:
        test_log = test_log.reshape(-1, 5)

    name = "_{}_{}x{}x{}_{}_{}.csv".format(
        info["game_name"],
        info["net_size"][0],
        info["net_size"][1],
        info["net_size"][2],
        info["evolution_method"],
        info["num_gene_samples"],
    )

    # save as csv, but add one row before the data to store the names of the columns (iteration,env_steps,reward_mean,reward_max)
    train_log = np.vstack(
        [
            ["iteration", "env_steps", "reward_mean", "reward_max", "max_reward_std"],
            train_log,
        ]
    )
    test_log = np.vstack(
        [
            ["iteration", "env_steps", "reward_mean", "reward_max", "max_reward_std"],
            test_log,
        ]
    )
    pd.DataFrame(train_log).to_csv(
        os.path.join(path, "train_log" + name), header=False, index=False
    )
    pd.DataFrame(test_log).to_csv(
        os.path.join(path, "test_log" + name), header=False, index=False
    )


def load_checkpoint(path):
    """
    Load a training checkpoint from a given path.

    Args:
        path (str): The directory path to load the checkpoint from.

    Returns:
        tuple: A tuple containing the search distribution, test results,
               metadata, training log, and testing log.
    """
    args_path = os.path.join(path, "args.txt")
    search_dist_path = os.path.join(path, "search_dist.save")
    tests_path = os.path.join(path, "tests.save")

    info = None

    with open(args_path, "rb") as f:
        info = json.load(f)

    name = "_{}_{}x{}x{}_{}_{}.csv".format(
        info["game_name"],
        info["net_size"][0],
        info["net_size"][1],
        info["net_size"][2],
        info["evolution_method"],
        info["num_gene_samples"],
    )

    search_dist = joblib.load(search_dist_path)
    tests = joblib.load(tests_path)

    # load train and test logs
    train_log = pd.read_csv(
        os.path.join(path, "train_log" + name), delimiter=","
    ).to_numpy()
    test_log = pd.read_csv(
        os.path.join(path, "test_log" + name), delimiter=","
    ).to_numpy()

    return search_dist, tests, info, train_log, test_log


def softmax(x):
    """
    Compute the softmax of an array.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The softmax of the input array.
    """
    max_rows = np.max(x, axis=0, keepdims=True)
    e_x = np.exp(x - max_rows)
    sum_rows = np.sum(e_x, axis=0, keepdims=True)
    f_x = e_x / sum_rows
    return f_x


def normalizeData(data, ascending=True, sum_to_one=False):
    """
    Normalize data to a range of [0, 1] or to sum to one.

    Args:
        data (np.ndarray): The data to be normalized.
        ascending (bool, optional): Whether ascending order is better. Defaults to True.
        sum_to_one (bool, optional): Whether to normalize the data to sum to one. Defaults to False.

    Returns:
        np.ndarray: The normalized data.
    """
    data = np.array(data)

    if not ascending:
        data = -data

    if sum_to_one:
        return softmax(data)
    else:
        if (np.max(data) - np.min(data)) == 0:
            return data
        else:
            return (data - np.min(data)) / (np.max(data) - np.min(data))


def getElites(scores, ascending, weights):
    """
    Return the sorted indices of elites according to the given criteria.

    Args:
        scores (list): A list of scores for each sample.
        ascending (list): A list of booleans indicating if lower scores are better.
        weights (list): The weights to apply to each score.

    Returns:
        tuple: A tuple containing the sorted indices and the total scores.
    """
    normalized_scores = np.zeros((len(scores), len(scores[0])))

    weights = normalizeData(weights, sum_to_one=True)

    for i in range(len(scores)):
        score = normalizeData(scores[i], ascending[i]) * weights[i]

        normalized_scores[i] = score

    total_score = np.sum(normalized_scores, axis=0)
    sorted_idxs = total_score.argsort()
    total_score = -total_score
    return sorted_idxs, total_score


def split_samples(samples, num_workers):
    """
    Split a list of samples into a number of batches for parallel processing.

    Args:
        samples (list): The list of samples to split.
        num_workers (int): The number of workers to split the samples for.

    Returns:
        np.ndarray: A reshaped array of samples.
    """
    samples = np.asarray(list(samples))
    return samples.reshape(num_workers, int(samples.shape[0] / num_workers))
