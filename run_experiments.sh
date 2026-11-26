#!/usr/bin/env bash
# Runs the experiments by executing the notebooks with papermill.
#
#   ./run_experiments.sh tune    step 1: learning rates (alpha=0.5) and FedProx mu (alpha=0.1), seed 11 only
#   ./run_experiments.sh grid    step 2: 4 methods x 4 data settings x 3 seeds on both datasets, using best_settings()
#
# Each run writes results/<name>.json (+ .pt), its executed notebook runs/<name>.ipynb and its output runs/<name>.log.
# Finished runs are skipped, so the script can simply be restarted after an interruption.
# DRY_RUN=1 ./run_experiments.sh grid   only prints the runs.
# JOBS=4 ./run_experiments.sh grid      runs 4 experiments at the same time on the same GPU (about 2 CPU cores each).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs results

JOBS=${JOBS:-1}
# split the CPU cores between the parallel runs, so they don't fight over threads
CORES=${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 4)}
export OMP_NUM_THREADS=$(( CORES / JOBS > 0 ? CORES / JOBS : 1 ))

DATASETS="femnist cifar"
SEEDS="11 12 13"
# Data settings: alphaX = Dirichlet label skew (0.1 = very skewed, 100 = close to IID),
#                c1     = pathological split, every client holds exactly 1 class
SETTINGS="alpha0.1 alpha0.5 alpha100 c1"
LOCAL_STEPS=50
TUNE_ROUNDS=50
GRID_ROUNDS=200

# Fill these in after the tune stage (see section 1 of summary.ipynb).
# FedProx and FedAdam use the SGD learning rate on the clients.
best_settings() {
  case $1 in
    femnist) SGD_LR=0.03; MOM_LR=0.01; ADAM_SERVER_LR=0.01; PROX_MU=0.01 ;;
    cifar)   SGD_LR=0.03; MOM_LR=0.01; ADAM_SERVER_LR=0.01; PROX_MU=0.01 ;;
  esac
}

notebook_for() {
  case $1 in
    femnist) echo dfml-project-new.ipynb ;;
    cifar)   echo dfml-project-cifar.ipynb ;;
  esac
}

num_classes() {
  case $1 in
    femnist) echo 62 ;;
    cifar)   echo 10 ;;
  esac
}

# papermill arguments for a data setting: alphaX -> Dirichlet with 10 clients,
# cN -> pathological with N classes per client and one client per class (62 for FEMNIST, 10 for CIFAR-10)
partition_args() {
  case $2 in
    alpha*) echo "-p partition dirichlet -p alpha ${2#alpha} -p num_clients 10" ;;
    c*)     echo "-p partition pathological -p classes_per_client ${2#c} -p num_clients $(num_classes "$1")" ;;
  esac
}

# run <dataset> <stage> <setting> <client_optimizer> <server_optimizer> <client_lr> <server_lr> <mu> <seed> <rounds>
run() {
  local name="$1_$2_$3_$4_$5_clr$6_slr$7_mu$8_seed$9"
  if grep -qs '"finished": true' "results/$name.json"; then
    echo "skip $name (already finished)"; return
  fi
  echo "run  $name"
  if [ "${DRY_RUN:-0}" = 1 ]; then return; fi
  # wait until fewer than JOBS runs are active (a failed run does not stop the others)
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done
  # shellcheck disable=SC2046  # partition_args is meant to split into separate arguments
  papermill "$(notebook_for "$1")" "runs/$name.ipynb" $(partition_args "$1" "$3") \
    -p stage "$2" -p client_optimizer "$4" -p server_optimizer "$5" \
    -p client_lr "$6" -p server_lr "$7" -p mu "$8" -p SEED "$9" -p rounds "${10}" \
    -p local_steps "$LOCAL_STEPS" -p run_name "$name" > "runs/$name.log" 2>&1 &
}

# download both datasets once before starting runs in parallel (parallel first downloads can corrupt the cache)
if [ "${DRY_RUN:-0}" != 1 ]; then
  python -c "from datasets import load_dataset; load_dataset('flwrlabs/femnist'); load_dataset('uoft-cs/cifar10')"
fi

case "${1:-}" in
  tune)
    for d in $DATASETS; do
      for lr  in 0.01 0.03 0.1;        do run "$d" tune alpha0.5 sgd          sgd  "$lr" 0.01   0.01  11 $TUNE_ROUNDS; done  # FedAvg
      for lr  in 0.003 0.01 0.03;      do run "$d" tune alpha0.5 sgd_momentum sgd  "$lr" 0.01   0.01  11 $TUNE_ROUNDS; done  # FedAvg + momentum
      for slr in 0.003 0.01 0.03 0.1;  do run "$d" tune alpha0.5 sgd          adam 0.03  "$slr" 0.01  11 $TUNE_ROUNDS; done  # FedAdam
      for mu  in 0.001 0.01 0.1;       do run "$d" tune alpha0.1 fedprox      sgd  0.03  0.01   "$mu" 11 $TUNE_ROUNDS; done  # FedProx
    done ;;
  grid)
    for seed in $SEEDS; do            # seeds outermost: after the first pass you already have one full picture
      for d in $DATASETS; do
        best_settings "$d"
        for s in $SETTINGS; do
          run "$d" grid "$s" sgd          sgd  "$SGD_LR" 0.01              0.01       "$seed" $GRID_ROUNDS  # FedAvg
          run "$d" grid "$s" sgd_momentum sgd  "$MOM_LR" 0.01              0.01       "$seed" $GRID_ROUNDS  # FedAvg + momentum
          run "$d" grid "$s" fedprox      sgd  "$SGD_LR" 0.01              "$PROX_MU" "$seed" $GRID_ROUNDS  # FedProx
          run "$d" grid "$s" sgd          adam "$SGD_LR" "$ADAM_SERVER_LR" 0.01       "$seed" $GRID_ROUNDS  # FedAdam
        done
      done
    done ;;
  *)
    echo "usage: $0 tune|grid"; exit 1 ;;
esac

wait || true      # let the last runs finish
if [ "${DRY_RUN:-0}" != 1 ]; then
  unfinished=$(grep -L '"finished": true' results/*.json 2>/dev/null || true)
  if [ -n "$unfinished" ]; then
    echo "Not finished (see runs/<name>.log; rerun the script to retry):"; echo "$unfinished"
  fi
fi
