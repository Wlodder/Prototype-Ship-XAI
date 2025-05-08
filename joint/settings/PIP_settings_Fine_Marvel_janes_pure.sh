#!/bin/bash
# general settings
NUM_FEATURES=200
SEED=1234
BATCH_SIZE=8
BATCH_SIZE_PRETRAIN=8
NET="convnext_tiny_26"
EPOCHS_PRETRAIN=3
EPOCHS_TRAIN=30
SAVE_INTERVAL=2
EVAL_INTERVAL=10
VIS_INTERVAL=4
IMAGE_SIZE=256
K=15
BUFFER_SIZE=6
TYPE='folder'

SUBJECT_DIR=${NET}_${EPOCHS_PRETRAIN}_${EPOCHS_TRAIN}_${NUM_FEATURES}_${BATCH_SIZE}_${BATCH_SIZE_PRETRAIN}_${IMAGE_SIZE}_PIPNet/${JANES_EXPERIMENT_DIRECTORY}/${K}_${BUFFER_SIZE}_main

# Create log dir and image dir
LOGDIR=$PIPNET_PROTO_DIR/$SUBJECT_DIR/log
IMGDIR=$PIPNET_PROTO_DIR/$SUBJECT_DIR/img_dir
CHECKPOINT=$LOGDIR/checkpoints/net_trained_last

mkdir -p $LOGDIR
mkdir -p $IMGDIR

python $PIPNET_RUNDIR/pure.py --dataset "Janes_Military_Marvel" --net $NET --batch_size $BATCH_SIZE\
  --batch_size_pretrain $BATCH_SIZE_PRETRAIN --epochs $EPOCHS_TRAIN\
    --epochs_pretrain $EPOCHS_PRETRAIN --lr 0.025 --lr_net 0.0005 --lr_block 0.0005\
    --log_dir $LOGDIR --num_features $NUM_FEATURES --image_size $IMAGE_SIZE\
 --dir_for_saving_images $IMGDIR --seed 1234\
 --save_epoch_interval $SAVE_INTERVAL --eval_epoch_interval $EVAL_INTERVAL --visualize_epoch_interval $VIS_INTERVAL\
 --extra_test_image_folder $SUGGESTION_JANES_MARVEL_PATH --proto_dir $LOGDIR --num_workers 4\
 --buffer_type buffer:35 --k 45 --patch_size 32 --state_dict_dir_net $CHECKPOINT\
  --debug 
