#!/bin/bash

model_config_path="configs/model.yaml"
mode_config_path="configs/test.yaml"

model_name="NetVGG"
model_version='v1'

mode="test"

dataset_name="megadepth"
#dataset_name="aachen"
#dataset_name="hpatches_view"
#dataset_name="hpatches_illum"
#dataset_name-="hpatches_view,hpatches_illum"

gpu=2

rm "nohup.out"

nohup python3 ~/personal/Summertime/Net/run.py --model_config_path="${model_config_path}" --mode_config_path="${mode_config_path}" \
 --model_name="${model_name}" --model_version="${model_version}" --mode="${mode}" --dataset_name="${dataset_name}" --gpu="${gpu}"&