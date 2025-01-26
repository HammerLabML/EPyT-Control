[![pypi](https://img.shields.io/pypi/v/epyt-control.svg)](https://pypi.org/project/epyt-control/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/epyt-control)
[![Documentation Status](https://readthedocs.org/projects/epyt-control/badge/?version=stable)](https://epyt-control.readthedocs.io/en/stable/?badge=stable)
[![Downloads](https://static.pepy.tech/badge/epyt-control)](https://pepy.tech/project/epyt-control)
[![Downloads](https://static.pepy.tech/badge/epyt-control/month)](https://pepy.tech/project/epyt-control)

# EPyT-Control -- EPANET Python Toolkit - Control

EPyT-Control is a Python package building on top of [EPyT-Flow](https://github.com/WaterFutures/EPyT-Flow) 
for implementing and evaluating control algorithms & strategies in water distribution networks (WDNs).

A special focus of this Python package is Reinforcement Learning for data-driven control in WDNs and
therefore it provides full compatibility with the
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) package.


## Installation

EPyT-Control supports Python 3.9 - 3.13

### PyPI

```
pip install epyt-control
```

### Git
Download or clone the repository:
```
git clone https://github.com/WaterFutures/EPyT-Control.git
cd EPyT-Control
```

Install all requirements as listed in [REQUIREMENTS.txt](REQUIREMENTS.txt):
```
pip install -r REQUIREMENTS.txt
```

Install the toolbox:
```
pip install .
```


## Quick Example

#### Interface of Environments

Basic example demonstrating the environments' interface:

```python
# Define/Specify MyEnv
# ....

# Load hypothetical environment "MyEnv"
with MyEnv() as env:
    # Show the observation space
    print(f"Observation space: {env.observation_space}")

    # Run 1000 iterations -- assuming that autorest=True
    obs, info = env.reset()
    for _ in range(1000):
        # Sample and apply a random action from the action space.
        # TODO: Replace with some smart RL/control method
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action)

        # Show action and observed reward
        print(action, reward)
```

#### Applying Reinforcement Learning to a given Environment

Simple example of using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) for learning a policy to control the chlorine injection in a given environment called ```SimpleChlorineInjectionEnv```:

```python
from stable_baselines3 import PPO
from gymnasium.wrappers import NormalizeObservation

# Define/Specify SimpleChlorineInjectionEnv
# ....

# Load chlorine injection environment
with SimpleChlorineInjectionEnv() as env:
    # Wrap environment
    env = NormalizeObservation(env)

    # Apply a simple policy learner
    # You might want to add more wrappers (e.g. normalizing inputs, rewards, etc.) and logging here
    # Also, inceasing the number of time steps might help as well
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("my_model_clinject.zip")  # Save policy
```

## Documentation

Documentation is available on readthedocs: [https://epyt-control.readthedocs.io/en/stable/](https://epyt-control.readthedocs.io/en/stable)

## License

MIT license -- see [LICENSE](LICENSE)

## How to Cite?

If you use this software, please cite it as follows:

```
@misc{github:epytcontrol,
        author = {André Artelt},
        title = {{EPyT-Control -- EPANET Python Toolkit - Control}},
        year = {2024},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/WaterFutures/EPyT-Control}}
    }
```

## How to get Support?

If you come across any bug or need assistance please feel free to open a new
[issue](https://github.com/WaterFutures/EPyT-Control/issues/)
if non of the existing issues answers your questions.

## How to Contribute?

Contributions (e.g. creating issues, pull-requests, etc.) are welcome --
please make sure to read the [code of conduct](CODE_OF_CONDUCT.md) and
follow the [developers' guidelines](DEVELOPERS.md).
