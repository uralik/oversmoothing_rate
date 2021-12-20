#!/bin/bash
#SBATCH --job-name=iwslt17fr
#SBATCH --open-mode=append
#SBATCH --output=./slurm_output/%A_%a_%j_%x.out
#SBATCH --error=./slurm_output/%A_%a_%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=1-105

PYTHON_SWEEP_PATH=../sweep_configs/iwslt17_finetune_oversmoothing.py
EXPERIMENT_NAME=finetune_iwslt17_oversmoothing_grid_label_smoothing
LANGUAGE=fr
FAIRSEQ_MODULE=../fairseq_module

singularity exec --nv --overlay $SCRATCH/mae9785_overlay.ext3:ro /scratch/mae9785/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate nmt

python  ${PYTHON_SWEEP_PATH} --call_fn ${EXPERIMENT_NAME} --sweep_step ${SLURM_ARRAY_TASK_ID} --language ${LANGUAGE} | xargs python ${FAIRSEQ_MODULE}/train.py

python ${PYTHON_SWEEP_PATH}  --call_fn validate_trained_sweep_ontest --experiment_name_to_validate ${EXPERIMENT_NAME} --sweep_step ${SLURM_ARRAY_TASK_ID} --beam 5 --language ${LANGUAGE} | xargs python ${FAIRSEQ_MODULE}/validate.py

python ${PYTHON_SWEEP_PATH}  --call_fn validate_trained_sweep_ontest --experiment_name_to_validate ${EXPERIMENT_NAME} --sweep_step ${SLURM_ARRAY_TASK_ID} --beam 1000 --language ${LANGUAGE} | xargs python ${FAIRSEQ_MODULE}/validate.py

python ${PYTHON_SWEEP_PATH}  --call_fn validate_trained_sweep_ontest --experiment_name_to_validate ${EXPERIMENT_NAME} --sweep_step ${SLURM_ARRAY_TASK_ID} --beam 100 --language ${LANGUAGE} | xargs python ${FAIRSEQ_MODULE}/validate.py

python ${PYTHON_SWEEP_PATH}  --call_fn validate_trained_sweep_ontest --experiment_name_to_validate ${EXPERIMENT_NAME} --sweep_step ${SLURM_ARRAY_TASK_ID} --beam 250 --language ${LANGUAGE} | xargs python ${FAIRSEQ_MODULE}/validate.py

python ${PYTHON_SWEEP_PATH}  --call_fn validate_trained_sweep_ontest --experiment_name_to_validate ${EXPERIMENT_NAME} --sweep_step ${SLURM_ARRAY_TASK_ID} --beam 500 --language ${LANGUAGE} | xargs python ${FAIRSEQ_MODULE}/validate.py

python ${PYTHON_SWEEP_PATH}  --call_fn validate_trained_sweep_ontest --experiment_name_to_validate ${EXPERIMENT_NAME} --sweep_step ${SLURM_ARRAY_TASK_ID} --beam 750 --language ${LANGUAGE} | xargs python ${FAIRSEQ_MODULE}/validate.py

python ${PYTHON_SWEEP_PATH}  --call_fn validate_trained_sweep_ontest --experiment_name_to_validate ${EXPERIMENT_NAME} --sweep_step ${SLURM_ARRAY_TASK_ID} --beam 1 --language ${LANGUAGE} | xargs python ${FAIRSEQ_MODULE}/validate.py
"
