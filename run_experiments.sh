#!/usr/bin/env bash
# Runs the experiments by executing the notebooks with papermill.
#
#   ./run_experiments.sh tune    step 1: client learning rate of FedAvg and FedAvg + momentum
#                                -> fill SGD_LR and MOM_LR in best_settings()
#   ./run_experiments.sh tune2   step 2: every method gets its own client learning rate, tuned jointly with its
#                                own setting: FedAdam client lr x server lr, FedProx client lr x mu
#                                -> fill ADAM_CLIENT_LR, ADAM_SERVER_LR, PROX_LR and PROX_MU
#   ./run_experiments.sh grid    step 3: 4 methods x 4 data settings x 3 seeds on both datasets
#
# Tuning uses seed 11 and runs every configuration at alpha=0.5 AND alpha=0.1. Each gets its own best values
# (summary.ipynb, section 1): "mild" = best at alpha=0.5, used for alpha0.5 and alpha100;
#                             "strong" = best at alpha=0.1, used for alpha0.1 and c1.
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
CLIENT_LRS="0.01 0.03 0.1 0.3"     # client learning rates tried for FedAvg, FedProx and FedAdam
TUNE_SETTINGS="alpha0.1 alpha0.5"  # every tuning configuration runs on both

# which tuned values a data setting uses
regime() {
  case $1 in
    alpha0.5|alpha100) echo mild ;;     # tuned at alpha=0.5
    alpha0.1|c1)       echo strong ;;   # tuned at alpha=0.1
  esac
}

# Best values from the tuning steps (see section 1 of summary.ipynb). Empty = not chosen yet.
# Every method has its own client learning rate.
best_settings() {   # best_settings <dataset> <mild|strong>
  case "$1 $2" in
    # alpha=0.5: FedAvg lr 0.01/0.03/0.1/0.3 -> FEMNIST 61.6/75.2/80.6/81.6, CIFAR 52.2/58.0/61.6/58.3
    #            momentum lr 0.003/0.01/0.03/0.1 -> FEMNIST 72.0/79.4/82.1/81.1, CIFAR 56.2/63.4/diverged
    "femnist mild")   SGD_LR=0.3; MOM_LR=0.03; PROX_LR=""; PROX_MU=""; ADAM_CLIENT_LR=""; ADAM_SERVER_LR="" ;;
    "femnist strong") SGD_LR="";  MOM_LR="";   PROX_LR=""; PROX_MU=""; ADAM_CLIENT_LR=""; ADAM_SERVER_LR="" ;;
    "cifar mild")     SGD_LR=0.1; MOM_LR=0.01; PROX_LR=""; PROX_MU=""; ADAM_CLIENT_LR=""; ADAM_SERVER_LR="" ;;
    "cifar strong")   SGD_LR="";  MOM_LR="";   PROX_LR=""; PROX_MU=""; ADAM_CLIENT_LR=""; ADAM_SERVER_LR="" ;;
  esac
}

# stop with a clear message if a value in best_settings() is still empty
require() {
  local where=$1; shift
  for v in "$@"; do
    if [ -z "${!v}" ]; then echo "Fill in $v for \"$where\" in best_settings() first."; exit 1; fi
  done
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
      case $d in
        femnist) MOM_LRS="0.003 0.01 0.03 0.1" ;;
        cifar)   MOM_LRS="0.003 0.01 0.03" ;;                                           # 0.03 already diverges
      esac
      for s in $TUNE_SETTINGS; do
        for lr in $CLIENT_LRS; do
          run "$d" tune "$s" sgd sgd "$lr" 0.01 0.01 11 $TUNE_ROUNDS                    # FedAvg
        done
        for lr in $MOM_LRS; do
          run "$d" tune "$s" sgd_momentum sgd "$lr" 0.01 0.01 11 $TUNE_ROUNDS           # FedAvg + momentum
        done
      done
    done ;;
  tune2)
    for d in $DATASETS; do
      for s in $TUNE_SETTINGS; do
        for clr in $CLIENT_LRS; do
          for slr in 0.003 0.01 0.03 0.1; do
            run "$d" tune "$s" sgd adam "$clr" "$slr" 0.01 11 $TUNE_ROUNDS              # FedAdam
          done
          for mu in 0.001 0.01 0.1 1; do
            run "$d" tune "$s" fedprox sgd "$clr" 0.01 "$mu" 11 $TUNE_ROUNDS            # FedProx
          done
        done
      done
    done ;;
  grid)
    for d in $DATASETS; do
      for r in mild strong; do
        best_settings "$d" "$r"; require "$d $r" SGD_LR MOM_LR PROX_LR PROX_MU ADAM_CLIENT_LR ADAM_SERVER_LR
      done
    done
    for seed in $SEEDS; do            # seeds outermost: after the first pass you already have one full picture
      for d in $DATASETS; do
        for s in $SETTINGS; do
          best_settings "$d" "$(regime "$s")"
          run "$d" grid "$s" sgd          sgd  "$SGD_LR" 0.01              0.01       "$seed" $GRID_ROUNDS  # FedAvg
          run "$d" grid "$s" sgd_momentum sgd  "$MOM_LR" 0.01              0.01       "$seed" $GRID_ROUNDS  # FedAvg + momentum
          run "$d" grid "$s" fedprox      sgd  "$PROX_LR"        0.01              "$PROX_MU" "$seed" $GRID_ROUNDS  # FedProx
          run "$d" grid "$s" sgd          adam "$ADAM_CLIENT_LR" "$ADAM_SERVER_LR" 0.01       "$seed" $GRID_ROUNDS  # FedAdam
        done
      done
    done ;;
  *)
    echo "usage: $0 tune|tune2|grid"; exit 1 ;;
esac

wait || true      # let the last runs finish
if [ "${DRY_RUN:-0}" != 1 ]; then
  unfinished=$(grep -L '"finished": true' results/*.json 2>/dev/null || true)
  if [ -n "$unfinished" ]; then
    echo "Not finished (see runs/<name>.log; rerun the script to retry):"; echo "$unfinished"
  fi
fi
