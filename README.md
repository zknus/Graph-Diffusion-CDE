To reproduce our results in Table 2:
First download the dataset in https://github.com/heterophily-submit/HeterophilousDatasets
Change the data path in line297 of ./src/data.py to your path of the downloaded datasets
Download the dataset in https://github.com/SitaoLuan/ACM-GNN/tree/main/new_data
Then run:

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
python run_GNN_raw.py --dataset workers --function belconv --time 3 --epoch 1000 --step_size 1 --dropout 0.2 --lr 0.01 --method euler --no_early --random_split --cuda 3 --hidden_dim 64 --block attention
run_GNN_raw.py --dataset workers --function gatconv --time 1 --epoch 1000 --step_size 1 --dropout 0.2 --lr 0.01 --method euler --no_early --cuda 2 --hidden_dim 64 --block constant --decay 0.001

for the full parameters used, please refer to ./best_log folder

