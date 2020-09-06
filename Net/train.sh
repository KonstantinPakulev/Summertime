#!/bin/bash

model_config_path="configs/model.yaml"
mode_config_path="configs/train.yaml"

model_name="NetVGG"
model_version='v4'

mode="train"

gpu=0
#gpu=1
#gpu=2

rm "nohup.out"

nohup python3 ~/personal/Summertime/Net/run.py --model_config_path="${model_config_path}" --mode_config_path="${mode_config_path}" \
 --model_name="${model_name}" --model_version="${model_version}" --mode="${mode}" --gpu="${gpu}"&