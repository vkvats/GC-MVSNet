#!/usr/bin/env bash
TESTPATH="/nfs/jolteon/data/ssd/vkvats/datasets/DTU/Testing"
TESTLIST="lists/dtu/test.txt"
TEST_DATASET="dtu" # dtu tnt
OUT_DIR='./outputs/stable_deform_Trans_CE_8srcInConsAvgGeo_b1'
CHECKPOINTS="12" ## run in batch of 4 checkpoints
photo_threshold=0.2
NVIEWS=5


python3 dynamic_fusion.py \
--testpath=$TESTPATH \
--testlist=$TESTLIST \
--test_dataset=$TEST_DATASET \
--thres_view=$NVIEWS \
--outdir=$OUT_DIR \
--checkpoints=$CHECKPOINTS \
--photo_threshold=$photo_threshold