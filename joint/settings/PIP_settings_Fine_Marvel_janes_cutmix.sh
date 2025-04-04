#!/bin/bash
# general settings
NUM_FEATURES=200
SEED=1234
BATCH_SIZE=16
BATCH_SIZE_PRETRAIN=16
NET="convnext_tiny_26"
EPOCHS_PRETRAIN=3
EPOCHS_TRAIN=30
SAVE_INTERVAL=2
EVAL_INTERVAL=10
VIS_INTERVAL=5
IMAGE_SIZE=300

SUBJECT_DIR=${NET}_${EPOCHS_PRETRAIN}_${EPOCHS_TRAIN}_${NUM_FEATURES}_${BATCH_SIZE}_${BATCH_SIZE_PRETRAIN}_${IMAGE_SIZE}_Janes

# Create log dir and image dir
LOGDIR=$PIPNET_PROTO_DIR/$SUBJECT_DIR/log
IMGDIR=$PIPNET_PROTO_DIR/$SUBJECT_DIR/img_dir

mkdir -p $LOGDIR
mkdir -p $IMGDIR

python $PIPNET_RUNDIR/main_cutmix.py --dataset "Janes_Military_Marvel" --net $NET --batch_size $BATCH_SIZE\
  --batch_size_pretrain $BATCH_SIZE_PRETRAIN --epochs $EPOCHS_TRAIN\
    --epochs_pretrain $EPOCHS_PRETRAIN --lr 0.05 --lr_net 0.0005 --lr_block 0.0005\
    --log_dir $LOGDIR --num_features $NUM_FEATURES --image_size $IMAGE_SIZE\
 --dir_for_saving_images $IMGDIR --seed 1234\
 --save_epoch_interval $SAVE_INTERVAL --eval_epoch_interval $EVAL_INTERVAL --visualize_epoch_interval $VIS_INTERVAL\
 --extra_test_image_folder $SUGGESTION_JANES_MARVEL_PATH --proto_dir $LOGDIR
