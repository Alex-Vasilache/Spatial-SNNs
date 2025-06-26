from src.task import Task
import numpy as np
import os
from util.args import dotdict
from render_browser import render_browser
from util.data import load_checkpoint
from pyvirtualdisplay.display import Display
import time
import gymnasium as gym
from pygame.time import Clock
import cv2
from src.search.evolution import Gene

TICK_LEN = 33


class Visualizer:
    """
    Handles the visualization of trained agents in the gym environment.

    This class loads a trained model, sets up the environment for rendering,
    and provides a method to run and visualize the agent's performance.
    """

    def __init__(self, path):
        """
        Initializes the Visualizer.

        Args:
            path (str): The path to the directory containing the training data and model checkpoints.
        """
        self.data_path = path

    def get_ith_best_elite(self, i):
        """
        Retrieves the i-th best elite from the saved population.

        Args:
            i (int): The rank of the elite to retrieve (e.g., 0 for the best).

        Returns:
            tuple: A tuple containing the Gene of the i-th best elite and its score.
        """
        elites, scores, _ = self.tests
        elites, scores = (
            self.search_dist.elites,
            self.search_dist.scores,
        )
        sort_idx = np.argsort(scores)[::-1]
        scores = scores[sort_idx]
        elites = np.array(elites)[sort_idx]
        return self.search_dist.elites[0], self.search_dist.scores[0]

    def initialize_viz(self):
        """
        Initializes the visualization environment.

        This method loads the necessary data from the checkpoint, sets up the
        gym environment with RGB array rendering, and prepares the task for visualization.
        """
        while not os.path.exists(os.path.join(self.data_path, "args.txt")):
            time.sleep(1)

        search_dist, tests, args, _, _ = load_checkpoint(self.data_path)
        # convert list of lists to 2 dimensional numpy array
        self.search_dist = search_dist
        self.tests = tests
        args = dotdict(args)
        args.batch_size_data = 1
        args.batch_size_gene = 1
        args.num_data_samples = 1
        args.num_gene_samples = 1

        args.data_path = self.data_path
        args.visualization = True
        args.test = True
        self.task = Task(args)
        if args.game_name is None:
            raise ValueError("Game name cannot be None")
        self.task.envs = np.array(
            [gym.make(args.game_name, render_mode="rgb_array")]
        )
        self.clock = Clock()

        self.params = self.get_ith_best_elite(1)[0]
        self.iteration = args.weights_iteration
        if "mujoco" in self.task.envs[0].spec.entry_point:
            _display = Display(visible=False, size=(800, 800))
            _ = _display.start()
        self.game_name = args.game_name

    def render(self, score, iteration, actions=None):
        """
        Renders a single frame of the environment with overlay text.

        This method captures the current frame from the environment, adds text
        to display the score and iteration number, and optionally displays the
        actions taken by the agent.

        Args:
            score (float): The current score to display.
            iteration (int): The current iteration number to display.
            actions (np.ndarray, optional): The actions taken by the agent. Defaults to None.

        Returns:
            np.ndarray: The rendered frame as a NumPy array.
        """
        img = self.task.envs[0].render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # get average color of image in the top half
        avg_color = np.mean(img[: int(img.shape[0] / 2)], axis=(0, 1))
        # get font color
        font_color = (255, 255, 255) if np.mean(avg_color) < 127 else (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Score: " + str(int(score))

        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        # get coords based on boundary
        textX = int((img.shape[1] - textsize[0]) / 2)
        textY = int(textsize[1] * 1.1)
        img = cv2.putText(
            img, text, (textX, textY), font, 1, font_color, 2, cv2.LINE_AA
        )

        text = "Iteration: " + str(int(iteration))
        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        # get coords based on boundary
        textX = int((img.shape[1] - textsize[0]) / 2)
        textY = int(textsize[1] * 3)
        prev_textY = int(textsize[1] * 4)
        img = cv2.putText(
            img, text, (textX, textY), font, 1, font_color, 2, cv2.LINE_AA
        )

        # render actions
        if actions is not None:
            text = ""
            if self.task.continuous_action_space:
                text += " ".join(["{:.2f}".format(action) for action in actions[0]])
            else:
                text += " ".join(["{:.2f}".format(action) for action in actions[0]])

            textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
            textX = int((img.shape[1] - textsize[0]) / 2)
            textY = prev_textY + int(textsize[1])

            # check if text is out of bounds
            if textsize[0] > img.shape[1]:
                # split text into required number of lines
                # determine number of lines
                n_lines = int(textsize[0] / img.shape[1]) + 1
                line_len = int(len(text) / n_lines)
                # split text into lines
                text = [text[i : i + line_len] for i in range(0, len(text), line_len)]
            else:
                text = [text]

            for i, line in enumerate(text):
                textsize = cv2.getTextSize(line, font, 0.5, 1)[0]
                textX = int((img.shape[1] - textsize[0]) / 2)
                textY = prev_textY + int(textsize[1] * 1.1) * i
                img = cv2.putText(
                    img, line, (textX, textY), font, 0.5, font_color, 1, cv2.LINE_AA
                )

        return img

    @render_browser
    def start(self):
        """
        Starts the visualization loop.

        This method runs the agent in the environment, renders the frames,
        and yields them for display in a browser. It also handles reloading
        the model and resetting the environment when an episode ends.
        """
        loop_done = False
        self.initialize_viz()
        self.task.set_params(
            [self.params] if isinstance(self.params, Gene) else self.params
        )
        assert self.task.args is not None
        states = self.task._reset(np.array([self.task.args.random_seed]))
        scores = np.zeros(1)
        ticks = np.zeros(1)
        dones = [False]
        video_frames = []

        img = self.render(scores[0], self.iteration)
        video_frames.append(img)
        self.clock.tick(TICK_LEN)
        yield img

        while not loop_done:
            try:
                assert self.task.net is not None
                states, rewards, dones, truncs, actions = self.task._step(states, dones)
                scores += rewards
                ticks += 1

                img = self.render(scores[0], self.iteration, actions)
                video_frames.append(img)

                if dones[0] or ticks[0] > 2000:
                    pool, args = None, None
                    try:
                        search_dist, tests, args, _, _ = load_checkpoint(self.data_path)
                        self.search_dist = search_dist
                        self.tests = tests
                    except EOFError:
                        assert self.task.net is not None
                        self.task.net.stop()
                        self.task.set_params(self.params)
                        dones = [False]
                        assert self.task.args is not None
                        if self.task.args.random_seed is not None:
                            self.task.args.random_seed += 1
                        states = self.task._reset(
                            np.array([self.task.args.random_seed])
                        )
                        scores = np.zeros(1)
                        ticks = np.zeros(1)
                        self.save_video(
                            video_frames,
                            os.path.join(self.data_path, self.task.game_name + ".mp4"),
                        )
                        video_frames = []
                        continue

                    args = dotdict(args)
                    args.batch_size = 1
                    args.pool_size = 1
                    args.device = "cpu"

                    if args.weights_iteration != self.iteration:
                        self.params = self.get_ith_best_elite(1)[0]
                        self.iteration = args.weights_iteration
                        # self.task.set_params(self.params)

                    assert self.task.net is not None
                    self.task.net.stop()
                    self.task.set_params(self.params)
                    dones = [False]
                    assert self.task.args is not None
                    if self.task.args.random_seed is not None:
                        self.task.args.random_seed += 1
                    states = self.task._reset(np.array([self.task.args.random_seed]))
                    scores = np.zeros(1)
                    ticks = np.zeros(1)
                    self.save_video(
                        video_frames,
                        os.path.join(self.data_path, self.task.game_name + ".mp4"),
                    )
                    video_frames = []

                self.clock.tick(TICK_LEN)
                yield img
            except KeyboardInterrupt:
                assert self.task.net is not None
                self.task.net.stop()
                loop_done = True

    def save_video(self, frames, path):
        """
        Saves a sequence of frames as a video file.

        Args:
            frames (list): A list of frames (NumPy arrays) to be saved.
            path (str): The path to save the video file.
        """
        height, width, layers = frames[0].shape
        size = (width, height)

        out = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*"mp4v"), 30, size)
        for i in range(len(frames)):
            out.write(frames[i])
        out.release()

        # write self.map_scores to txt
        # np.savetxt(
        #     path.split(".")[0] + ".txt",
        #     np.round(
        #         np.array(self.map_scores)
        #         .reshape(self.map_scores.shape[0], 10, -1)
        #         .mean(-1),
        #         0,
        #     ),
        #     fmt="%4.0f",
        # )
