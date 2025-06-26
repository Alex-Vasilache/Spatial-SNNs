# Evolving Spatially Embedded Recurrent Spiking Neural Networks for Control Tasks

This repository contains the official implementation for the paper "Evolving Spatially Embedded Recurrent Spiking Neural Networks for Control Tasks."

The code implements a framework for evolving Spiking Neural Networks (SNNs) for continuous control tasks. The core of this work is the use of a genetic algorithm to optimize SNNs that are spatially embedded in a 3D Euclidean space, where connection probabilities and strengths are influenced by the distance between neurons.

This framework can be used to replicate the experiments presented in the paper and to serve as a basis for further research into designing efficient, spatially embedded SNNs for neuromorphic control.

## Installation

To get started, first install the required system packages, then set up a Python virtual environment and install the Python dependencies.

### System Dependencies

Install the necessary system packages:

```bash
sudo apt-get update
sudo apt-get install -y xvfb ffmpeg libsm6 libxext6 software-properties-common curl
```

### Python Environment Setup

Install Python 3.8 and create a virtual environment:

```bash
# Add the deadsnakes PPA for access to different Python versions
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update

# Install Python 3.8 and related packages
sudo apt-get install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv

# Install pip for Python 3.8
curl https://bootstrap.pypa.io/pip/3.8/get-pip.py | sudo python3.8

# Create and activate a virtual environment
python3.8 -m venv norse-env
source norse-env/bin/activate
```

### Python Dependencies

Clone the repository and install the required Python dependencies:

```bash
pip install -r requirements.txt --ignore-installed
```

### Tested Environment

This code has been tested and verified to work in the following environment:

- **Operating System:** Ubuntu 20.04 LTS
- **Python Version:** Python 3.8
- **Hardware:** CUDA-compatible GPU (recommended for training)

While the code may work on other systems, these specifications represent the confirmed working environment.

## Usage

### Training

To train a new model, run the `wrapper.py` script. You can specify various command-line arguments to configure the training process.

```bash
python wrapper.py
```

For a full list of available arguments, please refer to `util/args.py`.

### Testing

To tast and visualize an already trained agent, you need to provide the path to the checkpoint directory.

```bash
python wrapper.py --test True --checkpoint_path "./trained/hopper-2d"
```

This will start a web server for visualization. You can view the agent's performance by navigating to the provided URL in your web browser.

#### Example Training Configuration

Here is an example of a more complex training configuration. Below the command is a detailed explanation of each argument.

```bash
python wrapper.py \
    --game_name CartPole-v1 \
    --discretize_intervals 0 \
    --net_size 8 8 1 \
    --num_gene_samples 100 \
    --batch_size_gene 100 \
    --spike_steps 4 \
    --num_iterations 1000 \
    --batch_size_data 3 \
    --num_data_samples 3 \
    --visualization True \
    --random_seed 0 \
    --max_vthr 1000 \
    --max_env_steps 1000 \
    --curiculum_learning False \
    --prune_unconnected True \
    --evolution_method classic
```

**Argument Explanations:**

| Argument | Value | Description |
| :--- | :--- | :--- |
| `game_name` | `CartPole-v1` | The name of the Gymnasium environment to use. |
| `discretize_intervals` | `0` | Disables discretization for continuous action spaces, treating them as continuous. |
| `net_size` | `8 8 1` | Defines the dimensions of the hidden neuron grid (8x8x1). |
| `num_gene_samples` | `100` | The total population size for each generation. |
| `batch_size_gene` | `100` | Number of networks to evaluate in parallel on the GPU. |
| `spike_steps` | `4` | Number of simulation steps for the SNN per environment step. |
| `num_iterations` | `1000` | The total number of generations to run the evolution for. |
| `batch_size_data` | `3` | Number of parallel environments for evaluating each network. |
| `num_data_samples` | `3` | Total number of evaluation rollouts for each network. |
| `visualization` | `True` | Enables live visualization of the agent's performance. |
| `random_seed` | `0` | The random seed for reproducibility. |
| `max_vthr` | `1000` | Maximum membrane potential threshold for the LIF neurons. |
| `max_env_steps` | `1000` | Maximum number of steps per episode. |
| `curiculum_learning`| `False` | Disables curriculum learning (gradual difficulty increase). |
| `prune_unconnected` | `True` | Enables pruning of unconnected neurons. |
| `evolution_method` | `classic` | Uses a classic evolutionary algorithm (as opposed to MAP-Elites). |

## Publication

This work, "Evolving Spatially Embedded Recurrent Spiking Neural Networks for Control Tasks," by Alexandru Vasilache, Jona Scholz, Yulia Sandamirskaya, and Jürgen Becker, has been accepted for publication and will be presented at the 34th International Conference on Artificial Neural Networks (ICANN 2025).

The full paper is available here: [PDF](https://drive.google.com/file/d/1V5Si801bhVcBfmPYKX1BwtXfi4HO-NY4/view?usp=sharing)

If you use this code in your research, please consider citing our paper.

## Demonstration

A version of the code in this repository was adapted to evolve a neuromorphic controller for a physical inverted pendulum. The following video demonstrates the resulting SNN controller successfully performing the swing-up and balancing task in a real-world setting:

[Neuromorphic Control: Inverted Pendulum Swing-Up & Balancing](https://www.youtube.com/watch?v=Y0yKGLlRkW4&ab_channel=AlexandruVasilache)

---

## Troubleshooting

Some common installation issues and their solutions are listed below.
<details>

### Xvfb Not Found

If you encounter a `FileNotFoundError` for `Xvfb`, you can install it using the following command:

```bash
sudo apt-get install xvfb -y
```

### OpenGL Error

If you see an error related to `libGL.so.1`, you can resolve it by installing the necessary dependencies:

```bash
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 -y
```

### Debugger Connection Error (`pydevd`)

If you encounter debugger connection errors during a debugging session, such as `OSError: [Errno 9] Bad file descriptor`, this may be caused by an underlying issue with the rendering environment, particularly `Xvfb`. The traceback may look similar to this:

```
OSError: [Errno 9] Bad file descriptor
...
pydevd: Sending message related to process being replaced timed-out after 5 seconds
```

This error can occur if the headless display server (`Xvfb`) or its dependencies are not installed correctly, causing the main process to crash and breaking the debugger's connection.

To resolve this, ensure that both `xvfb` and its related graphics libraries are installed:

```bash
sudo apt-get update
sudo apt-get install -y xvfb ffmpeg libsm6 libxext6
```

### Port in Use

```
Starting rendering, check `server_ip:5000`.
 * Serving Flask app 'render_browser.render'
 * Debug mode: off
Address already in use
Port 5000 is in use by another program. Either identify and stop that program, or start the server with a different port.
```

Go to `/usr/local/lib/python3.8/dist-packages/render_browser/render.py`

Replace file with:

```python
import os
import cv2
from flask import Flask, request, render_template, Response

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(DIR_PATH, 'templates/')

app = Flask(__name__, template_folder=TEMPLATE_PATH)

@app.route('/')
def index():
    return render_template('index.html')

def frame_gen(env_func, *args, **kwargs):
    get_frame = env_func(*args, **kwargs)
    while True:
        frame = next(get_frame, None)
        if frame is None:
            break
        _, frame = cv2.imencode('.png', frame)
        frame = frame.tobytes()
        yield (b'--frame\r\n' + b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

def render_browser(env_func):
    def wrapper(*args, **kwargs):
        @app.route('/render_feed')
        def render_feed():
            return Response(frame_gen(env_func, *args, **kwargs), mimetype='multipart/x-mixed-replace; boundary=frame')

        #print("Starting rendering, check `server_ip:5000`.")
        port = 5000
        suc = False
        while not suc:
            try:
                app.run(host='0.0.0.0', port=str(port), debug=False)
                suc = True
            except:
                port += 1

    return wrapper

if __name__ == '__main__':
    print("Testing Gym Browser Render")
    import gym

    @render_browser
    def random_policy():
        env = gym.make('Breakout-v0')
        env.reset()

        for _ in range(100):
            yield env.render(mode='rgb_array')
            action = env.action_space.sample()
            env.step(action)

    random_policy()
```

### `norse_op.so` Error

If you encounter an error similar to `ImportError: /usr/local/lib/python3.10/dist-packages/norse_op.so: undefined symbol: ...`, it is likely due to an incompatibility between `norse` and your PyTorch/Python versions. This issue can be resolved by switching to Python 3.8.

Follow these steps to install Python 3.8 and set it as the default version:

```bash
# Install packages required to add new repositories and download files
sudo apt-get install -y software-properties-common curl

# Add the deadsnakes PPA for access to different Python versions
sudo add-apt-repository ppa:deadsnakes/ppa -y

# Update package lists and install Python 3.8
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-dev python3.8-distutils

# Install pip for Python 3.8
curl https://bootstrap.pypa.io/pip/3.8/get-pip.py | sudo python3.8

# Configure python alternatives to select a default version.
# This gives python3.10 a higher priority, so you will need to manually select python3.8.
# Note: replace python3.10 with your current version if it's different.
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2

# Select Python 3.8 from the interactive prompt
sudo update-alternatives --config python3

# In the interactive prompt that follows, select the number corresponding to Python 3.8.

# Finally, reinstall the project dependencies
pip install -r requirements.txt --ignore-installed
```

</details>

## License and Copyright

Copyright © 2025 Alexandru Vasilache

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Citing This Work

If you use this code in your research, please cite:

```bibtex
@inproceedings{vasilache2025evolving,
  title={Evolving Spatially Embedded Recurrent Spiking Neural Networks for Control Tasks},
  author={Vasilache, Alexandru and Scholz, Jona and Sandamirskaya, Yulia and Becker, J{\"u}rgen},
  booktitle={34th International Conference on Artificial Neural Networks (ICANN 2025)},
  year={2025}
}
```

### Contact

For questions or issues regarding this code, please contact:
- Alexandru Vasilache: vasilache@fzi.de