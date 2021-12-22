# Characterizing and addressing the issue of oversmoothing in neural autoregressive sequence modeling

This repo contains code for the paper [Characterizing and addressing the issue of oversmoothing in neural autoregressive sequence modeling](https://arxiv.org/pdf/2112.08914.pdf)

**Authors**: Ilia Kulikov, Maksim Eremeev, Kyunghyun Cho

![Oversmoothing rate plot](/oversmoothing_rate.png)

## Requirements

See the reqs.txt

## Environment variables

In order to utilize the scripts provided, one should set the following environment variables:

```bash
export OVERLAY_PATH= # only for SLURM + singularity
export CUDA_SIF_PATH= # only for SLURM + singularity

export EXPERIMENTS_DIRECTORY_WMT=
export EXPERIMENTS_DIRECTORY_IWSLT=

export DATA_IWSTL=
export DATA_WMT19_DEEN=
export DATA_WMT19_RUEN=
export DATA_WMT19_ENDE=
export DATA_WMT16_ENDE=

export PRETRAINED_MODEL_WMT19_DEEN=
export PRETRAINED_MODEL_WMT19_RUEN=
export PRETRAINED_MODEL_WMT19_ENDE=
export PRETRAINED_MODEL_WMT16_ENDE=

export FAIRSEQ_MODULE=

export RESULTS_DIRECTORY=
```

## Installing fairseq

Please use the fairseq with the following commit to avoid issues with updated codebase:

`git clone https://github.com/pytorch/fairseq.git`

`git checkout c6006678261bf5d52e2c744508b5ddd306cafebd`

Install fairseq:

```
cd fairseq
pip install --editable ./
```

## Running experiments from the paper

We use SLURM manager to run all experiments in this work. We use Nvidia RTX8000 gpus to train and validate models. We provide slurm scripts which use singularity containers, each of them has such block:

```
python  ${PYTHON_SWEEP_PATH} --call_fn ${EXPERIMENT_NAME} --sweep_step ${SLURM_ARRAY_TASK_ID} --language ${LANGUAGE} | xargs python ${FAIRSEQ_MODULE}/train.py
```

Where the first python command calls the slurm sweep factory which generates command line arguments. These arguments are then fed into the fairseq process. If you do not use slurm or singularity, then you can directly use these commands.

### Downloading prertained checkpoints for WMT tasks

WMT16 En-De: https://github.com/pytorch/fairseq/tree/main/examples/scaling_nmt#pre-trained-models

WMT19 Models (single models before fine-tuning): https://github.com/pytorch/fairseq/tree/main/examples/wmt19#pre-trained-single-models-before-finetuning 

Note that dict files and bpe codes from WMt19 models are used as part of data preparation scripts later.

### Step 1. Data preparation

We tested the proposed approach on the following datasets: IWSLT'17 {DE, FR, ZH}-EN, WMT'19 {RU, DE}-EN, WMT'19 EN-DE, WMT'16 EN-DE.
You can find bash scripts for data preprocessing in the `/data` directory.

At the time of publication of this work the google drive link to WMT16 En-De preprocessed data from Google became invalid.

Please note: the BPE codes for WMT tasks come with the pretrained checkpoints. Please download checkpoints first.

**fairseq preprocessing**: we use preprocessing command from fairseq which takes in the dictionary files in case of WMT tasks where pretrained checkpoints come with corresponding dictionaries.

### Step 2. Model training

Model configurations we used during the experiments can be found in the `/sweep_configs` directory. Each file contains experiment setups for a specific dataset. In order to train the model use

```bash
python sweep_configs/CONFIG_SCRIPT --call_fn EXPERIMENT_NAME --sweep_step CONFIG_SERIAL_NUMBER (--language SOURCE_LANGUAGE) | xargs python fairseq_module/train.py
```

You can find more examples in the `/slurm_files` directory. Note that `--language` parameter only applies to IWSLT experiments. `CONFIG_SERIAL_NUMBER` defines the step which maps to a specific configuration to run.

### Step 3. Model validation

We prepared custom validation script to compute and store all nessessary statistics in a pickle file. In the post-processing step we parse these files to compuite average statistics across training runs. To run the validation, execute the following command:

```bash
python sweep_configs/CONFIG_SCRIPT  --call_fn validate_trained_sweep_ontest --experiment_name_to_validate EXPERIMENT_NAME --sweep_step CONFIG_SERIAL_NUMBER --beam BEAM_SIZE (--language SOURCE_LANGUAGE) | xargs fairseq_module/validate.py
```

You can find more examples in the `/slurm_files` directory. Note that `--language` parameter only applies to IWSLT experiments.

### Step 4. Collecting the results

`/parsers` directory contains several tools that automatically process the validation pickles. 

The following command executes the post-processing:

```bash
python parsers/process_extra_state_pickles.py -p EXPERIMENT_RESULTS_PATH -b BEAM_SIZE -o OUTPUT_DIRECTORY
```

You can find more examples by looking at `/parsers/submit_jobs*` slurm files..

### Step 5. Visualizing the result

Run `/experiments.ipynb` notebook to render the plots that are used in the paper.


## BibTex

If you want to cite our work, please use the following bib:

```
@misc{kulikov2021characterizing,
      title={Characterizing and addressing the issue of oversmoothing in neural autoregressive sequence modeling}, 
      author={Ilia Kulikov and Maksim Eremeev and Kyunghyun Cho},
      year={2021},
      eprint={2112.08914},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
