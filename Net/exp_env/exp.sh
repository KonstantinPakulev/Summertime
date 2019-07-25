#!/usr/bin/env bash

time="$(date "+%m_%d_%H_%M_%S")"
log_dir="$(pwd)/runs/"${time}""
checkpoint_dir="$(pwd)/checkpoints/"${time}""

mkdir ${log_dir}
mkdir ${checkpoint_dir}

echo "Log dir: ${log_dir}"
echo "Checkpoints dir:: ${checkpoint_dir}"

stdout=${log_dir}"/out.out"
stderr=${log_dir}"/out.err"

python3 /home/konstantin/PycharmProjects/Summertime/Net/exp_env/exp.py --log_dir=${log_dir}