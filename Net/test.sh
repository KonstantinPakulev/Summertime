#!/usr/bin/env bash

exp_id=${1}
exp_name=${2}
checkpoint_iter=${3}

log_dir="$(pwd)/runs/"${exp_name}
checkpoint_dir="$(pwd)/checkpoints/"${exp_name}

stdout=${log_dir}"/test_out.out"
stderr=${log_dir}"/test_out.err"

rm -f "${stdout}"
rm -f "${stderr}"

bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o "$stdout" -e "$stderr" python ~/Summertime/Net/run.py --exp_id="${exp_id}" --log_dir="${log_dir}" --checkpoint_dir="${checkpoint_dir}" --checkpoint_iter="${checkpoint_iter}"