# nmt_eos

journal: https://hackmd.io/@yhHA_s_ZSzOr_a6yKdSwWg/SkLeZj7Nd/edit

Preparing the data: 

https://github.com/pytorch/fairseq/tree/master/examples/translation#iwslt14-german-to-english-transformer

Example usage:

training:

```
python ./fairseq_module/train.py   data-bin/iwslt14.tokenized.de-en.eostask/  --task translation_eos   --arch transformer_iwslt14_deen_meos     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000     --dropout 0.3 --weight-decay 0.0001     --criterion cross_entropy     --max-tokens 4096     --eval-bleu     --eval-bleu-args '{"beam": 10, "max_len_a": 1.2, "max_len_b": 10, "min_length": 0}'     --eval-bleu-detok moses     --eval-bleu-remove-bpe     --eval-bleu-print-samples     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --user-dir ./fairseq_module/ --number_eos_tokens 10 --save-dir ./debug/ --eos-choice max
```

validation (python command wrapped in bash script):

```
bash validate_script.sh ./debug/
```

Notes:

* Extra criterion `cross_entropy_eos` replicates `cross_entropy` and dumps `task.train_batch_statistics` with some running statistics on every `train_step` which we may need for research.