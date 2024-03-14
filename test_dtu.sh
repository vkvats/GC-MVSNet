#!/usr/bin/env bash
TESTPATH="/nfs/jolteon/data/ssd/vkvats/datasets/DTU/Testing" 
TESTLIST="lists/dtu/test.txt"
FUSIBLE_PATH="/u/vkvats/tynamo/fusibile/fusibile" 	# path to fusible of fusibile
CKPT_FILE="checkpoints/def_8S_InConsAvg12_b3_DI284_G7/model_000009.ckpt"
OUTDIR="outputs/testing"
DEVICE_ID=2
max_h=864 # 864, 1024
max_w=1152 # 1152, 1280
## Gipuma filter paramerters
FILTER_METHOD="gipuma" ## normal, gipuma
gipuma_prob_thresh=0.0
gipuma_disparity_thresh=0.09
gipuma_num_consistenc=2
depth_interval_ratio="1.6,0.7,0.3"
ndepths="48,32,8" 

if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi


python3 test.py \
--dataset=general_eval \
--batch_size=1 \
--filter_method=$FILTER_METHOD \
--prob_threshold=$gipuma_prob_thresh \
--disp_threshold=$gipuma_disparity_thresh \
--num_consistent=$gipuma_num_consistenc \
--max_h=$max_h \
--max_w=$max_w \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt=$CKPT_FILE \
--outdir=$OUTDIR \
--device_id=$DEVICE_ID \
--numdepth=192 \
--ndepths=$ndepths \
--depth_inter_r=$depth_interval_ratio \
--interval_scale=1.06 \
--fusibile_exe_path=$FUSIBLE_PATH