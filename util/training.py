import tqdm


class ProgressBar:
    """
    A simple progress bar for tracking training iterations.

    This class wraps `tqdm` to provide a consistent progress bar format
    for both the main iteration loop and for displaying scalar metrics.
    """

    def __init__(self, num_iterations, verbose=True):
        """
        Initializes the ProgressBar.

        Args:
            num_iterations (int): The total number of iterations.
            verbose (bool, optional): Whether to display the progress bar. Defaults to True.
        """
        if verbose:  # create a nice little progress bar
            self.scalar_tracker = tqdm.tqdm(
                total=num_iterations,
                desc="Scalars",
                bar_format="{desc}",
                position=0,
                leave=True,
            )
            progress_bar_format = (
                "{desc} {n_fmt:"
                + str(len(str(num_iterations)))
                + "}/{total_fmt}|{bar}|{elapsed}<{remaining}"
            )
            self.progress_bar = tqdm.tqdm(
                total=num_iterations,
                desc="Iteration",
                bar_format=progress_bar_format,
                position=1,
                leave=True,
            )
        else:
            self.scalar_tracker = None
            self.progress_bar = None

    def __call__(self, _steps=1, **kwargs):
        """
        Updates the progress bar with the latest metrics.

        Args:
            _steps (int, optional): The number of steps to advance the progress bar. Defaults to 1.
            **kwargs: A dictionary of scalar metrics to display.
        """
        if self.progress_bar is not None:
            if self.scalar_tracker is not None:
                formatted_scalars = {
                    key: "{:.3e}".format(value[-1] if isinstance(value, list) else value)
                    for key, value in kwargs.items()
                }
                description = (
                    "Rewards: "
                    + "".join(
                        [
                            str(key) + "=" + value + ", "
                            for key, value in formatted_scalars.items()
                        ]
                    )
                )[:-2]
                self.scalar_tracker.set_description(description)
            self.progress_bar.update(_steps)


