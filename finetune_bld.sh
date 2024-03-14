#!/usr/bin/env bash
# run this script in the root path of TransMVSNet
MVS_TRAINING="/l/vision/zekrom_ssd/joshisri/MVSDatasets/BlendedMVS/" # path to dataset mvs_training
LOG_DIR="./checkpoints/bld/testing" # path to checkpoints
MASK_TYPE="joint_inconsistency_mask" ## choices: joint_inconsistency_mask, geo_consistency_to_inconsistency 
CONS2INCO_TYPE="average" ## Choices: average, normal, inverse, normal_and_average
AVERAGE_WEIGHT_GAP="0.1" ## 0.1: Avg weight [1,2,0.1], 0.2: Avg weight [1,3,0.2]
OPERATION="product"
nviews=7
GEO_MASK_SUM_TH=10 # min val: NVIEWS-1, Max val: 10 src views
R_DEPTH_MIN_THRESH="0.01,0.005,0.0025"
DIST_THRESH="1,0.5,0.25"
LR=0.0001
LREPOCHS="8,12,16:2"
weight_decay=0.001
EPOCHS=16
NGPUS=7
BATCH_SIZE=2
D=128
ndepths="48,32,8"
DLOSSW="1.0,1.0,2.0"
depth_interval_ratio="2.0,0.8,0.4" 
#"4.0,1.0,0.5" "2.0,0.8,0.4" "2.0,0.5,0.25" "1.4,0.5,0.25" "1.0,0.5,0.25" "1.6,0.8.0.4"
dtu_ckpt="./checkpoints/def_8S_InConsAvg12_b3_DI284_G7/gcmvsnet_dtu.ckpt"

if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi

python3 -m torch.distributed.launch --nproc_per_node=$NGPUS finetune.py \
	--logdir=$LOG_DIR \
    --loadckpt=$dtu_ckpt \
    --mask_type=$MASK_TYPE \
    --cons2incon_type=$CONS2INCO_TYPE \
	--avg_weight_gap=$AVERAGE_WEIGHT_GAP \
    --operation=$OPERATION \
	--dataset=bld_train \
	--batch_size=$BATCH_SIZE \
	--epochs=$EPOCHS \
	--trainpath=$MVS_TRAINING \
	--trainlist=lists/bld/training_list.txt \
	--testlist=lists/bld/validation_list.txt \
	--numdepth=$D \
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

