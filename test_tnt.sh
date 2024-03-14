#!/usr/bin/env bash
TESTPATH="/l/vision/zekrom_hdd/joshisri/cvlab_work/processed_tannksandtemples/intermediate/" # advanced, intermediate
TESTLIST="lists/tnt/inter.txt" # inter.txt, adv.txt
CKPT_FILE="./checkpoints/bld/def_8S_InConsAvg12_b3_DI284_G7_D128_N7/model_000014.ckpt"
OUT_DIR="/nfs/jolteon/data/ssd2/vkvats/TnT/def_8S_InConsAvg12_b3_DI284_G7_D128_N7/14_DIR"
DEVICE_ID=6
FILTER_METHOD="dynamic"
depth_interval_ratio="1.6,0.7,0.3" #training  = "2.0,0.8,0.4", "1.6,0.7,0.3"
D=192
ndepths="64,32,8"
filter="dynamic"
interval_scale=1.06
batch=1
nviews=11
## dynamic fusion specific
prob_confidence=0.0
thresh_view=10


# python3 test.py --dataset=tnt_eval \
#     --outdir=$OUT_DIR \
#     --num_view=$nviews \
#     --batch_size=$batch \
#     --interval_scale=$interval_scale \
#     --numdepth=$D \
#     --ndepths=$ndepths \
#     --depth_inter_r=$depth_interval_ratio \
#     --device_id=$DEVICE_ID \
#     --testpath=$TESTPATH  \
#     --testlist=$TESTLIST \
#     --filter_method=$filter \
#     --loadckpt $CKPT_FILE ${@:2}

   
# Filter method
python3 dynamic_fusion.py \
--testpath=$OUT_DIR \
--tntpath=$TESTPATH \
--testlist=$TESTLIST \
--outdir=$OUT_DIR \
--photo_threshold=$prob_confidence \
--thres_view=$thresh_view \
--test_dataset=tnt
