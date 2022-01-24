#!/bin/bash
#SBATCH --job-name=process_sweeps
#SBATCH --open-mode=append
#SBATCH --output=./slurm_output/%j_%x.out
#SBATCH --error=./slurm_output/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2:00:00
#SBATCH --mem=150G
#SBATCH -c 4
#SBATCH --array=1,5,100,250,500,750,1000

singularity exec --nv --overlay $OVERLAY_PATH:ro $CUDA_SIF_PATH /bin/bash -c "
source /ext3/env.sh
conda activate nmt

python process_extra_state_pickles.py -p ${EXPERIMENTS_DIRECTORY_IWSLT}/finetune_iwslt17_oversmoothing_grid_label_smoothing_zh -b ${SLURM_ARRAY_TASK_ID} -o ${RESULTS_DIRECTORY}
"
