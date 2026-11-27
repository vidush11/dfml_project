# dfml_project
Course project for Distributed and Federated Machine Learning (CSE665A, IIIT Delhi):
federated optimization (FedAvg, client momentum, FedProx, FedAdam) under non-IID client data on FEMNIST and CIFAR-10.

## Layout
- `dfml-project-new.ipynb`: FEMNIST (Dirichlet partitioning, `client` / `server` classes, training)
- `dfml-project-cifar.ipynb`: CIFAR-10, same code with a 3-block CNN with GroupNorm
- `run_experiments.sh`: runs the notebooks with papermill for every setting
- `summary.ipynb`: tables (mean ± std over seeds) and figures from the results
- `paper/`: LaTeX source of the report
- `notes/`: notes on the experiments

Everything the runs produce is written **outside the repo**, to `../runs/` (change with `OUT_DIR=...`),
so `git pull` never conflicts with results:
- `../runs/results/`: one `<run>.json` per run (config, test accuracy/loss, train loss, client drift, time per round) and `<run>.pt`
- `../runs/logs/`: executed notebook and output log of every run
- `../runs/figures/`: tables and figures from `summary.ipynb`

## Running
```bash
pip install -r requirements.txt          # install PyTorch for your CUDA version first
./run_experiments.sh tune                # 1. client learning rates of FedAvg and FedAvg + momentum (seed 11)
#    summary.ipynb section 1 -> fill SGD_LR and MOM_LR in best_settings() in run_experiments.sh
./run_experiments.sh tune2               # 2. FedAdam: client lr x server lr; FedProx: client lr x mu (each method its own client lr)
#    summary.ipynb section 1 -> fill PROX_LR, PROX_MU, ADAM_CLIENT_LR and ADAM_SERVER_LR
./run_experiments.sh grid                # 3. 4 methods x {alpha 0.1, 0.5, 100, 1 class per client} x 3 seeds x 2 datasets
#    open summary.ipynb for the tables and figures (saved to ../runs/figures/)
```
`JOBS=4 ./run_experiments.sh grid` runs 4 experiments at the same time on one GPU (give the job about 2 CPU cores per run).
`DRY_RUN=1 ./run_experiments.sh grid` lists the runs without starting them. Finished runs are skipped, so an interrupted
script can simply be started again. A single notebook can also be run by hand: edit the parameters cell at the top.
