#!/usr/bin/env bash

export NUM_FOLDS=256
export NUM_FOLDS_VAL=8
MAX_PARALLEL=32
ITERS=$(expr ${NUM_FOLDS} / ${MAX_PARALLEL})

# Training

mkdir -p logs
for i in $(seq 0 $((${ITERS}-1)))
do
  printf "\n\n\nStarting fold ${i}\n\n\n"
  parallel -j $(nproc --all) --will-cite "python3 prep_data.py -fold {1} -num_folds ${NUM_FOLDS} -split=train> logs/trainlog{1}.txt" ::: $(seq $((${MAX_PARALLEL} * $i)) $((${MAX_PARALLEL} * $i + ${MAX_PARALLEL} - 1)))
done

printf "\n\n\nStarting val\n\n\n"
parallel -j $(nproc --all) --will-cite "python3 prep_data.py -fold {1} -num_folds ${NUM_FOLDS_VAL} -split=val > logs/vallog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS_VAL}-1)))

printf "\n\n\nStarting test\n\n\n"
parallel -j $(nproc --all) --will-cite "python3 prep_data.py -fold {1} -num_folds ${NUM_FOLDS_VAL} -split=test > logs/testlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS_VAL}-1)))
