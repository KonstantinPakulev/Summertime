#!/usr/bin/env bash

colmap_path="/usr/bin"
base_path="/home/konstantin/datasets/MegaDepthDataset"

python3 undistor_reconstructions.py --colmap_path="${colmap_path}" --base_path="${base_path}"