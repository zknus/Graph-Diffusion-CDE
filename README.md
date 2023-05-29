# Graph Neural Convection-Diffusion with Heterophily

This repository contains the code for our IJCAI 2023 accepted paper, *[Graph Neural Convection-Diffusion with Heterophily](https://arxiv.org/abs/2305.16780)*. 

## Table of Contents

- [Requirements](#requirements)
- [Datasets](#datasets)
- [Reproducing Results](#reproducing-results)
- [Reference](#reference)
- [Citation](#citation)

## Requirements

To install the required dependencies, refer to the environment.yaml file


<!-- ## Datasets

To reproduce our results in Table 2, you first need to download the datasets.

1. Download the datasets from the following repositories:
    - [HeterophilousDatasets](https://github.com/heterophily-submit/HeterophilousDatasets)
    - [ACM-GNN/new_data](https://github.com/SitaoLuan/ACM-GNN/tree/main/new_data)

2. Update the data path in line 297 of `./src/data.py` with the path to the downloaded datasets. -->

## Reproducing Results

To reproduce the results in Table 2, run the following commands:

```bash
python run_GNN_raw.py --dataset amazon-ratings --function belconv --time 1 --epoch 1000 --step_size 1 --dropout 0.2 --lr 0.01 --method euler --no_early --cuda 1 --hidden_dim 64 --block constant  

python run_GNN_raw.py --dataset amazon-ratings --function gatconv --time 1 --epoch 1000 --step_size 0.5 --dropout 0.2 --lr 0.01 --method euler --no_early --random_split --cuda 2 --hidden_dim 64

python run_GNN_raw.py --dataset minesweeper --function belconv --time 3 --epoch 1000 --step_size 1 --dropout 0.2 --lr 0.01 --method rk4 --no_early --cuda 1 --hidden_dim 64 --block attention --decay 0.001

python run_GNN_raw.py --dataset minesweeper --function gatconv --time 4 --epoch 600 --step_size 1 --dropout 0.2 --lr 0.01 --method rk4 --no_early --cuda 2 --hidden_dim 64 --block constant --decay 0.001

python run_GNN_raw.py --dataset questions --function belconv --time 1 --epoch 1000 --step_size 1 --dropout 0.2 --lr 0.01 --method euler --no_early --cuda 1 --hidden_dim 64 --block constant

python run_GNN_raw.py --dataset questions --function gatconv --time 3 --epoch 1000 --step_size 1 --dropout 0.2 --lr 0.01 --method euler --no_early --cuda 3

python run_GNN_raw.py --dataset roman-empire --function belconv --time 1 --epoch 1000 --step_size 1 --dropout 0.2 --lr 0.01 --method euler --no_early --cuda 1 --hidden_dim 256 --block constant

python run_GNN_raw.py --dataset roman-empire --function gatconv --time 3 --epoch 1000 --step_size 1 --dropout 0.2 --lr 0.01 --method euler --no_early --cuda 2 --hidden_dim 64 --block constant --decay 0.001

python run_GNN_raw.py --dataset wiki-cooc --function belconv --time 1 --epoch 1000 --step_size 1 --dropout 0.2 --lr 0.01 --method euler --no_early --cuda 1 --hidden_dim 64 --block constant

python run_GNN_raw.py --dataset wiki-cooc --function transconv --time 1 --epoch 1000 --step_size 1 --dropout 0.2 --lr 0.01 --method euler --no_early --cuda 1 --hidden_dim 64 --block attention --decay 0.001
```

## Reference 

Our code is developed based on the following repo:
https://github.com/twitter-research/graph-neural-pde



## Citation

If you find our helpful, consider to cite us:
```bash
@inproceedings{zhao2023graph,
  title={Graph neural convection-diffusion with heterophily},
  author={Zhao, K. and Kang, Q. and Song, Y. and She, R. and Wang, S. and Tay, W. P.},
  booktitle={Proc. International Joint Conference on Artificial Intelligence},
  year={2023},
  month={Aug},
  address={Macao, China}
}



