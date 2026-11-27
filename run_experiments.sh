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
# Everything a run produces goes OUTSIDE the repo (default ../runs), so git pull / fetch never conflicts with results:
#   ../runs/results/<name>.json (+ .pt)   ../runs/logs/<name>.ipynb and <name>.log
# Finished runs are skipped, so the script can simply be restarted after an interruption.
# DRY_RUN=1 ./run_experiments.sh grid   only prints the runs.
# JOBS=4 ./run_experiments.sh grid      runs 4 experiments at the same time on the same GPU (about 2 CPU cores each).
# SKIP='pattern' ./run_experiments.sh grid   skips runs whose name matches the pattern (grep -E), e.g. runs done elsewhere.
set -euo pipefail
cd "$(dirname "$0")"
OUT_DIR=${OUT_DIR:-../runs}
RESULTS="$OUT_DIR/results"
LOGS="$OUT_DIR/logs"
mkdir -p "$RESULTS" "$LOGS"

JOBS=${JOBS:-1}
# split the CPU cores between the parallel runs, so they don't fight over threads
CORES=${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
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
    # tune2, FEMNIST alpha=0.5: FedProx best clr 0.3 mu 0.1 (82.4); FedAdam best clr 0.1 slr 0.03 (81.6, clr 0.03: 81.5)
    "femnist mild")   SGD_LR=0.3; MOM_LR=0.03; PROX_LR=0.3; PROX_MU=0.1;   ADAM_CLIENT_LR=0.1;  ADAM_SERVER_LR=0.03 ;;
    # alpha=0.1: FedAvg lr 0.01/0.03/0.1 -> FEMNIST 53.2/69.1/76.9, CIFAR 48.0/53.7/53.8 (0.3 worst on both)
    #            momentum lr 0.003/0.01/0.03 -> FEMNIST 65.6/75.5/78.9, CIFAR 51.5/55.9/49.8
    # tune2, FEMNIST alpha=0.1: FedProx best clr 0.1 mu 0.001 (76.9, mu 0.01 also 76.9); FedAdam best clr 0.03 slr 0.03 (78.9)
    "femnist strong") SGD_LR=0.1; MOM_LR=0.03; PROX_LR=0.1; PROX_MU=0.001; ADAM_CLIENT_LR=0.03; ADAM_SERVER_LR=0.03 ;;
    # tune2, CIFAR alpha=0.5: FedProx best clr 0.1 mu 0.01 (60.2, mu 0.001: 60.0); FedAdam best clr 0.03 slr 0.01 (57.0)
    "cifar mild")     SGD_LR=0.1; MOM_LR=0.01; PROX_LR=0.1; PROX_MU=0.01;  ADAM_CLIENT_LR=0.03; ADAM_SERVER_LR=0.01 ;;
    # tune2, CIFAR alpha=0.1: FedProx best clr 0.1 mu 0.01 (54.2, mu 0.1: 54.0); FedAdam best clr 0.03 slr 0.01 (54.9)
    "cifar strong")   SGD_LR=0.1; MOM_LR=0.01; PROX_LR=0.1; PROX_MU=0.01;  ADAM_CLIENT_LR=0.03; ADAM_SERVER_LR=0.01 ;;
  esac
}

# true if the given best_settings() values are filled in; otherwise says which run waits for tuning
ready() {
  local label=$1; shift
  for v in "$@"; do
    if [ -z "${!v}" ]; then echo "wait $label (fill $v in best_settings() first)"; return 1; fi
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
  if grep -qs '"finished": true' "$RESULTS/$name.json"; then
    echo "skip $name (already finished)"; return
  fi
  if pgrep -f "$name.ipynb" > /dev/null; then
    echo "skip $name (running in another call of this script)"; return
  fi
  if [ -n "${SKIP:-}" ] && echo "$name" | grep -qE "$SKIP"; then
    echo "skip $name (matches SKIP, e.g. runs on another machine)"; return
  fi
  echo "run  $name"
  if [ "${DRY_RUN:-0}" = 1 ]; then return; fi
  # wait until fewer than JOBS runs of this user are active, also counting runs started by an earlier call
  # of this script (a failed run does not stop the others)
  while [ "$(pgrep -u "$(id -u)" -f papermill | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  # shellcheck disable=SC2046  # partition_args is meant to split into separate arguments
  papermill "$(notebook_for "$1")" "$LOGS/$name.ipynb" $(partition_args "$1" "$3") \
    -p stage "$2" -p client_optimizer "$4" -p server_optimizer "$5" \
    -p client_lr "$6" -p server_lr "$7" -p mu "$8" -p SEED "$9" -p rounds "${10}" \
    -p local_steps "$LOCAL_STEPS" -p run_name "$name" -p results_dir "$RESULTS" > "$LOGS/$name.log" 2>&1 &
}

# download both datasets once before starting runs in parallel (parallel first downloads can corrupt the cache)
if [ "${DRY_RUN:-0}" != 1 ]; then
  python3 -c "from datasets import load_dataset; load_dataset('flwrlabs/femnist'); load_dataset('uoft-cs/cifar10')"
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
    # methods whose values are not tuned yet are skipped ("wait ..."); run grid again after filling them in
    for seed in $SEEDS; do            # seeds outermost: after the first pass you already have one full picture
      for d in $DATASETS; do
        for s in $SETTINGS; do
          best_settings "$d" "$(regime "$s")"
          ready "$d $s seed$seed FedAvg"   SGD_LR &&
            run "$d" grid "$s" sgd          sgd  "$SGD_LR"         0.01              0.01       "$seed" $GRID_ROUNDS
          ready "$d $s seed$seed Momentum" MOM_LR &&
            run "$d" grid "$s" sgd_momentum sgd  "$MOM_LR"         0.01              0.01       "$seed" $GRID_ROUNDS
          ready "$d $s seed$seed FedProx"  PROX_LR PROX_MU &&
            run "$d" grid "$s" fedprox      sgd  "$PROX_LR"        0.01              "$PROX_MU" "$seed" $GRID_ROUNDS
          ready "$d $s seed$seed FedAdam"  ADAM_CLIENT_LR ADAM_SERVER_LR &&
            run "$d" grid "$s" sgd          adam "$ADAM_CLIENT_LR" "$ADAM_SERVER_LR" 0.01       "$seed" $GRID_ROUNDS
        done
      done
    done ;;
  *)
    echo "usage: $0 tune|tune2|grid"; exit 1 ;;
esac

wait || true      # let the last runs finish
if [ "${DRY_RUN:-0}" != 1 ]; then
  unfinished=$(grep -L '"finished": true' "$RESULTS"/*.json 2>/dev/null || true)
  if [ -n "$unfinished" ]; then
    echo "Not finished (see $LOGS/<name>.log; rerun the script to retry):"; echo "$unfinished"
  fi
fi
