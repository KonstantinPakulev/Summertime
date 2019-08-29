#!/usr/bin/env bash

exp_id="TrainExperimentDebug"
log_dir="$(pwd)/runs/debug_train"
checkpoint_dir="$(pwd)/checkpoints/debug_train"

rm -rf "${log_dir}"
rm -rf "${checkpoint_dir}"

mkdir "${log_dir}"
mkdir "${checkpoint_dir}"

echo "Log dir: ${log_dir}"
echo "Checkpoints dir:: ${checkpoint_dir}"

python3 "$(pwd)/run.py" --exp_id="${exp_id}" --log_dir="${log_dir}" --checkpoint_dir="${checkpoint_dir}"