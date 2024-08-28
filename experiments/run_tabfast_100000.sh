#!/bin/bash

cd ..
source "./scripts/DATASETS_TABLE$1.sh"

for dataset in "${DATASETS[@]}"; do
    echo "Running experiment for dataset: ${dataset}"
    cd TabZilla
    python tabzilla_data_preprocessing.py --dataset_name $dataset
    cd ..
    CUDA_VISIBLE_DEVICES=$2 python TabZilla/tabzilla_experiment.py \
        --experiment_config TabZilla/100000train_samples_experiment_config.yml \
        --dataset_dir "TabZilla/datasets/${dataset}" \
        --model_name tabfast \
        --wandb
done
cd experiments