#!/bin/bash

file_name=$1
config_file=$2
GPUS=${3:-"0"}

# dist train
#CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node=4 "${file_name}" --config-file "${config_file}"

# single train
CUDA_VISIBLE_DEVICES="${GPUS}" python "${file_name}" --config-file "${config_file}"
