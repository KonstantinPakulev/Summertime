#!/usr/bin/env bash

time="$(date "+%m_%d_%H_%M_%S")"
log_dir="$(pwd)/runs/"${time}
checkpoint_dir="$(pwd)/checkpoints/"${time}

mkdir "${log_dir}"
mkdir "${checkpoint_dir}"

echo "Log dir: ${log_dir}"
echo "Checkpoints dir:: ${checkpoint_dir}"

python3 ./train.py --log_dir="${log_dir}" --checkpoint_dir="${checkpoint_dir}" --config="test"