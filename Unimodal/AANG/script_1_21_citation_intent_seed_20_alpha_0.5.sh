#!/bin/bash
#SBATCH -p gpu_highmem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem 56GB
#SBATCH --time 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@gmail.com # TODO
#SBATCH --chdir=/work/sakter/AANG
#SBATCH --output=/work/sakter/AANG/logs/citation_intent_tf_idf_seed_20_max_alpha_0.5.out # TODO
#SBATCH --error=/work/sakter/AANG/logs/citation_intent_tf_idf_seed_20_max_alpha_0.5.err # TODO

eval "$(conda shell.bash hook)"
conda activate aang


#CHEMPROT

# python -m scripts.autoaux --prim-task-id chemprot --train_data_file=datasets/chemprot/train.jsonl --dev_data_file=datasets/chemprot/dev.jsonl --test_data_file=datasets/chemprot/test.jsonl --output_dir /work/sakter/AANG/autoaux_outputs/TAPT/chemprot/auxlr_fast/seed=0 --model_type roberta-base --model_name_or_path roberta-base  --tokenizer_name roberta-base --per_gpu_train_batch_size=8  --gradient_accumulation_steps=64 --do_train --learning_rate=0.0001 --block_size 512 --logging_steps 10000 --classf_lr=0.0001 --classf_patience 20 --num_train_epochs=150  --classifier_dropout=0.3 --overwrite_output_dir --classf_iter_batchsz=8 --classf_ft_lr 1e-6 --classf_max_seq_len 512 --seed=0  --classf_dev_wd=0.1 --classf_dev_lr=0.01 -searchspace-config=tapt -task-data=datasets/chemprot/train.txt -in-domain-data=datasets/chemprot/domain.10xTAPT.txt -num-config-samples=1 --dev_batch_sz=16 --eval_every 30 -prim-aux-lr=0.1 -auxiliaries-lr=1 --classf_warmup_frac 0.06 --classf_wd 0.1 --base_wd 0.01 --dev_fit_iters 10 -step-meta-every 3 -token_temp 0.5  --classf-metric accuracy -pure-transform

#ACL-ARC

python -m scripts.autoaux --prim-task-id citation_intent --train_data_file=datasets/citation_intent/train.jsonl --dev_data_file=datasets/citation_intent/dev.jsonl --test_data_file=datasets/citation_intent/test.jsonl --output_dir /work/sakter/AANG/autoaux_outputs/TAPT/citation_intent/auxlr_tam_tf_idf_act_max_0.5/seed=20 --model_type roberta-base --model_name_or_path roberta-base  --tokenizer_name roberta-base --per_gpu_train_batch_size=8  --gradient_accumulation_steps=64 --do_train --learning_rate=0.0001 --block_size 512 --logging_steps 10000 --classf_lr=0.0001 --classf_patience 20 --num_train_epochs=150  --classifier_dropout=0.3 --overwrite_output_dir --classf_iter_batchsz=8 --classf_ft_lr 1e-6 --classf_max_seq_len 512 --seed=0  --classf_dev_wd=0.1 --classf_dev_lr=0.01 -searchspace-config=tapt -task-data=datasets/citation_intent/train.txt -in-domain-data=datasets/citation_intent/domain.10xTAPT.txt -num-config-samples=1 --dev_batch_sz=16 --eval_every 30 -prim-aux-lr=0.1 -auxiliaries-lr=1 --classf_warmup_frac 0.06 --classf_wd 0.1 --base_wd 0.01 --dev_fit_iters 10 -step-meta-every 3 -token_temp 0.5  --classf-metric f1 -pure-transform --mask_selection_process max --num_seed 20 --has_alpha --alpha 0.5


