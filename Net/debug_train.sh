#!/usr/bin/env bash

exp_id="debug_train"
log_dir="$(pwd)/runs/"${exp_id}
checkpoint_dir="$(pwd)/checkpoints/"${exp_id}

rm -rf "${log_dir}"
rm -rf "${checkpoint_dir}"

mkdir "${log_dir}"
mkdir "${checkpoint_dir}"

echo "Log dir: ${log_dir}"
echo "Checkpoints dir:: ${checkpoint_dir}"

python3 "$(pwd)/train.py" --exp_id="${exp_id}" --log_dir="${log_dir}" --checkpoint_dir="${checkpoint_dir}"