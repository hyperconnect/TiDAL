# TiDAL: Learning Training Dynamics for Active Learning

<a href="http://www.youtube.com/watch?feature=player_embedded&v=UfM2A4doHvk" target="_blank"><img src="http://img.youtube.com/vi/UfM2A4doHvk/0.jpg" alt="presentation" width="360" height="270" border="10" /></a>

- Accepted to [ICCV 2023](https://iccv2023.thecvf.com/)
- https://arxiv.org/abs/2210.06788
- This repository is the official implementation by the authors.
- Watch our [poster](./poster.pdf), [slides](https://docs.google.com/presentation/d/1vIejNaskHYJwA4AYF9iP2a-G-hGoL_Mp/edit?usp=sharing&ouid=118399620942177943626&rtpof=true&sd=true), or the [5 minute video presentation of this paper](https://youtu.be/UfM2A4doHvk?feature=shared) for more details.

## Prerequisites:   
- Linux or macOS
- Python 3.10
- CPU compatible but NVIDIA GPU + CUDA CuDNN is highly recommended.

## Installation

Install required Python packages

```
pip install requirements.txt
```


## Running code

To train the model(s) and evaluate in the paper, run this command:

```train
python main.py -d cifar10 -i 1 -m TiDAL -q Entropy
```
