#!/usr/bin/env bash

exp_id=${1}
exp_name=${2}
log_dir="$(pwd)/runs/"${exp_name}
checkpoint_dir="$(pwd)/checkpoints/"${exp_name}

rm -rf "${log_dir}"
rm -rf "${checkpoint_dir}"

mkdir "${log_dir}"
mkdir "${checkpoint_dir}"

echo "Log dir: ${log_dir}"
echo "Checkpoints dir:: ${checkpoint_dir}"

stdout=${log_dir}"/out.out"
stderr=${log_dir}"/out.err"

bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o "$stdout" -e "$stderr" python ~/Summertime/Net/run.py --exp_id="${exp_id}" --log_dir="${log_dir}" --checkpoint_dir="${checkpoint_dir}"