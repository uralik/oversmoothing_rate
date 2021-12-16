#!/bin/bash
#SBATCH --job-name=process_sweeps
#SBATCH --open-mode=append
#SBATCH --output=./slurm_output/%j_%x.out
#SBATCH --error=./slurm_output/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH --array=1,5,100,250,500,750,1000

singularity exec --nv --overlay $SCRATCH/mae9785_overlay.ext3:ro /scratch/mae9785/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate nmt

python process_extra_state_pickles.py -p /scratch/ik1147/nmt_multiple_eos/nmt_eos/finetune_wmt19_deen_osl -b ${SLURM_ARRAY_TASK_ID} -o /scratch/mae9785/new_results
"
