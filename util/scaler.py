import gymnasium as gym
import tqdm
from joblib import Parallel, delayed
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os


WORKERS = 64
RUNS = 10000


def get_runs(env_name, num_steps, workers, print_updates=False):
    if print_updates:
        print("Generating " + str(num_steps) + " random runs...")

    def get_run_chunk():
        env = gym.make(env_name)
        runs = []
        for _ in tqdm.tqdm(range(int(num_steps / workers))):
            states = []
            state, _ = env.reset()
            done = False
            trunc = False
            while not (done or trunc):
                states.append(state)
                action = env.action_space.sample()
                state, _, done, trunc, _ = env.step(action)
            runs.append(np.array(states).T)
        return runs

    if workers > 1:
        with Parallel(n_jobs=-1) as parallel:
            results = parallel(delayed(get_run_chunk)() for _ in range(workers))
        if results is not None:
            runs = [
                item for sublist in results if sublist is not None for item in sublist
            ]
        else:
            runs = []
    else:
        runs = get_run_chunk()

    return runs


def get_path_from_dirs(dirs):
    path = os.getcwd()

    for dir_name in dirs:
        path = os.path.join(path, dir_name)
        if not os.path.exists(path):
            os.mkdir(path)

    return path


def _optimize_scaler(runs, print_updates):
    if print_updates:
        print("Fitting scaler...")
    concat_runs = np.concatenate(runs, axis=1).T
    upper = (
        np.concatenate([abs(concat_runs.min(axis=0)), concat_runs.max(axis=0)])
        .reshape(-1, concat_runs.shape[1])
        .max(axis=0)
    )
    lower = -upper
    concat_runs = np.append(concat_runs, upper.reshape(1, -1), axis=0)
    concat_runs = np.append(concat_runs, lower.reshape(1, -1), axis=0)
    scaler = MinMaxScaler().fit(concat_runs)
    return scaler


def create_scaler(env, workers=WORKERS, num_steps=RUNS, print_updates=False):
    assert env.spec is not None
    env_path = get_path_from_dirs(["data", "env_bounds", env.spec.id])
    scaler_path = os.path.join(env_path, "scaler.save")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        if print_updates:
            print("Loaded scaler at " + scaler_path)
        return scaler

    runs = get_runs(env.spec.id, num_steps * 10, workers=workers)
    scaler = _optimize_scaler(runs, print_updates)
    joblib.dump(scaler, scaler_path)
    if print_updates:
        print("Saved scaler at " + scaler_path)
    return scaler
