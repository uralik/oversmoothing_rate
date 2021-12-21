# Characterizing and addressing the issue of oversmoothing in neural autoregressive sequence modeling

This repo contains code for the paper [Characterizing and addressing the issue of oversmoothing in neural autoregressive sequence modeling](https://arxiv.org/pdf/2112.08914.pdf)

**Authors**: Ilia Kulikov, Maksim Eremeev

## Requirements

```
fairseq==1.0.0a0+1ef3d6a
sacrebleu==2.0.0
torch==1.10.0
sacremoses==0.0.45
numpy==1.19.5
pandas==1.3.3
matplotlib==3.4.3
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

### Downloading prertained checkpoints for WMT tasks

TODO

### Step 1. Data preparation

We tested the proposed approach on the following datasets: IWSLT'17 {DE, FR, ZH}-EN, WMT'19 {RU, DE}-EN, WMT'19 EN-DE, WMT'16 EN-DE.
You can find bash scripts for data preprocessing in the `./data` directory.

At the time of publication of this work the google drive link to WMT16 En-De preprocessed data became invalid.

Please note: the BPE codes for WMT tasks come with the pretrained checkpoints. Please download checkpoints first.

### Step 2. Model training

Model configurations we used during the experiments can be found in the `/sweep_configs` directory. Each file contains experiment setups for a specific dataset. In order to train the model use

```bash
python sweep_configs/CONFIG_SCRIPT --call_fn EXPERIMENT_NAME --sweep_step CONFIG_SERIAL_NUMBER (--language SOURCE_LANGUAGE) | xargs python fairseq_module/train.py
```

You can find more examples in the `/slurm_files` directory. `--language` parameter only applies to IWSLT experiments.


### Step 3. Model validation

Custom validation script we use computes and stores all nessessary statistics in a pickle file. In the post-processing step we get those statistics prepared for visualization. To execute the validation procedure use

```bash
python sweep_configs/CONFIG_SCRIPT  --call_fn validate_trained_sweep_ontest --experiment_name_to_validate EXPERIMENT_NAME --sweep_step CONFIG_SERIAL_NUMBER --beam BEAM_SIZE (--language SOURCE_LANGUAGE) | xargs fairseq_module/validate.py
```

You can find more examples in the `/slurm_files` directory. `--language` parameter only applies to IWSLT experiments.

### Step 4. Collecting the results

`/parsers` directory contains several tools that automatically process the validation pickles. 

The following command executes the post-processing:

```bash
python parsers/process_extra_state_pickles.py -p EXPERIMENT_RESULTS_PATH -b BEAM_SIZE -o OUTPUT_DIRECTORY
```

You can find more examples by investigating `submit_jobs*` slurm files in the `/parsers` folder.

### Step 5. Visualizing the result

Execute `experiments.ipynb` notebook to render the plots that were used in the paper.


## BibTex

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
