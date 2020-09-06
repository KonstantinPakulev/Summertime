#!/usr/bin/env bash

config_path="configs/analyze.yaml"
namespace="Net.source.experiments.net"
gpu=0
#gpu=1
#gpu=2

mode="analyze"
dataset="megadepth"

model_name="NetVGG"
model_suffix='v21'

rm "nohup.out"

nohup python3 ~/personal/Summertime/Net/run.py --config_path="${config_path}" --namespace="${namespace}" --gpu="${gpu}" \
    --mode="${mode}" --dataset="${dataset}" --model_name="${model_name}" --model_suffix="${model_suffix}"&