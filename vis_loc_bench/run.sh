#!/usr/bin/env bash

dataset_path="/home/konstantin/datasets/Aachen-Day-Night"
colmap_path="/usr/bin"
method_name="NetVGG"

rm -r "${dataset_path}/sparse-${method_name}-empty"
rm -r "${dataset_path}/sparse-${method_name}-database"
rm -r "${dataset_path}/sparse-${method_name}-final"
rm -r "${dataset_path}/sparse-${method_name}-final-txt"
rm "${dataset_path}/${method_name}.db"

rm "nohup.out"

nohup python3 ~/personal/Summertime/vis_loc_bench/reconstruction_pipeline.py  --dataset_path "${dataset_path}" \
    --colmap_path "${colmap_path}" --method_name "${method_name}"&