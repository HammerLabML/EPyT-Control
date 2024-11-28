[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# EPyT-Control -- EPANET Python Toolkit - Control

EPyT-Control is a Python package building on top of [EPyT-Flow](https://github.com/WaterFutures/EPyT-Flow) 
for implementing and evaluating control algorithms & strategies in water distribution networks (WDNs).

A special focus of this Python package is Reinforcement Learning for data-driven control in WDNs and
therefore it provides full compatibility with the
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) package.


## Installation

EPyT-Control supports Python 3.9 - 3.12

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

## Documentation

Documentation is available on readthedocs: [https://epyt-control.readthedocs.io/en/latest/](https://epyt-control.readthedocs.io/en/stable)

## License

MIT license -- see [LICENSE](LICENSE)

## How to Cite?

If you use this software, please cite it as follows:

```
@misc{github:epytflow,
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
