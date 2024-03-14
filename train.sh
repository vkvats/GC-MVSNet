#!/usr/bin/env bash
# run this script in the root path of TransMVSNet
MVS_TRAINING="/nfs/jolteon/data/ssd/vkvats/datasets/DTU/" # path to dataset mvs_training
LOG_DIR="./checkpoints/testing" # path to checkpoints
MASK_TYPE="joint_inconsistency_mask" ## choices: joint_inconsistency_mask, geo_consistency_to_inconsistency 
CONS2INCO_TYPE="average" ## Choices: average, normal, inverse, normal_and_average
AVERAGE_WEIGHT_GAP="0.2" ## 0.1: Avg weight [1,2,0.1], 0.2: Avg weight [1,3,0.2]
OPERATION="product"
nviews=3
GEO_MASK_SUM_TH=8 # min val: NVIEWS-1, Max val: 10 src views
R_DEPTH_MIN_THRESH="0.01,0.005,0.0025"
DIST_THRESH="1,0.5,0.25"
LR=0.001
LREPOCHS="8,12,16,20:2"
weight_decay=0.0001
EPOCHS=36
NGPUS=1
BATCH_SIZE=1
ndepths="48,32,8"
DLOSSW="1.0,1.0,2.0"
depth_interval_ratio="2.0,0.8,0.4" 
#"4.0,1.0,0.5" "2.0,0.8,0.4" "2.0,0.5,0.25" "1.4,0.5,0.25" "1.0,0.5,0.25" "1.6,0.8.0.4"
# pretrained_ckpt="./checkpoints/model_dtu.ckpt"

if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi

torchrun --nproc-per-node=$NGPUS train.py \
	--logdir=$LOG_DIR \
    --mask_type=$MASK_TYPE \
    --cons2incon_type=$CONS2INCO_TYPE \
	--avg_weight_gap=$AVERAGE_WEIGHT_GAP \
    --operation=$OPERATION \
	--dataset=dtu_yao \
	--batch_size=$BATCH_SIZE \
	--epochs=$EPOCHS \
	--trainpath=$MVS_TRAINING \
	--trainlist=lists/dtu/train.txt \
	--testlist=lists/dtu/val.txt \
	--numdepth=192 \
	--ndepths=$ndepths \
	--nviews=$nviews \
	--wd=$weight_decay \
	--depth_inter_r=$depth_interval_ratio \
	--lrepochs=$LREPOCHS \
	--lr=$LR \
	--geo_mask_sum_thresh=$GEO_MASK_SUM_TH \
	--dist_thresh=$DIST_THRESH \
    --relative_depth_diff_min_thresh=$R_DEPTH_MIN_THRESH \
	--dlossw=$DLOSSW | tee -a $LOG_DIR/log.txt

