#!/usr/bin/env bash

time="$(date "+%m_%d_%H_%M_%S")"
log_dir="$(pwd)/runs/"${time}
checkpoint_path="../checkpoints/07_31_18_50_03/my_model_6.pth"

mkdir "${log_dir}"

echo "Log dir: ${log_dir}"

python3 /home/konstantin/PycharmProjects/Summertime/Net/exp_env/exp.py --log_dir="${log_dir}" --checkpoint_path="${checkpoint_path}"