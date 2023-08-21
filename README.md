# TiDAL: Learning Training Dynamics for Active Learning

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
