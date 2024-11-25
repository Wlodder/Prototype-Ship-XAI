#!/bin/bash
# general settings
NUM_FEATURES=10
SEED=1234
BATCH_SIZE=64
BATCH_SIZE_PRETRAIN=64
NET="resnet50"

# Create log dir and image dir
LOGDIR=$PIPNET_PROTO_DIR/log
IMGDIR=$PIPNET_PROTO_DIR/img_dir

mkdir -p $LOGDIR
mkdir -p $IMGDIR

python $PIPNET_RUNDIR/main.py --dataset "Fine_Military_Marvel" --net $NET --batch_size $BATCH_SIZE\
  --batch_size_pretrain $BATCH_SIZE_PRETRAIN --epochs 10\
    --epochs_pretrain 30 --lr 0.05 --lr_net 0.0005 --lr_block 0.0005\
    --log_dir $LOGDIR --num_features $NUM_FEATURES --image_size 224\
 --dir_for_saving_images $IMGDIR --seed 1234
