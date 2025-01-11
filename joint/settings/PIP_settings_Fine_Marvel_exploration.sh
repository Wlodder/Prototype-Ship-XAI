#!/bin/bash
# general settings
NUM_FEATURES=400
SEED=1234
BATCH_SIZE=48
BATCH_SIZE_PRETRAIN=64
NET="convnext_tiny_26"
EPOCHS_PRETRAIN=3
EPOCHS_TRAIN=30
SAVE_INTERVAL=10
EVAL_INTERVAL=3

SUBJECT_DIR=${NET}_${EPOCHS_PRETRAIN}_${EPOCHS_TRAIN}_${NUM_FEATURES}
WEIGHTS_FILE=$1

# Create log dir and image dir
LOGDIR=$PIPNET_PROTO_DIR/exploration/$SUBJECT_DIR/$WEIGHTS_FILE/log
IMGDIR=$PIPNET_PROTO_DIR/exploration/$SUBJECT_DIR/$WEIGHTS_FILE/img_dir
NETDIR=$PIPNET_PROTO_DIR/$SUBJECT_DIR/log/checkpoints/$WEIGHTS_FILE

mkdir -p $LOGDIR
mkdir -p $IMGDIR

python $PIPNET_RUNDIR/exploration.py --dataset "Fine_Military_Marvel" --net $NET --batch_size $BATCH_SIZE\
  --batch_size_pretrain $BATCH_SIZE_PRETRAIN --epochs $EPOCHS_TRAIN\
    --epochs_pretrain $EPOCHS_PRETRAIN --lr 0.05 --lr_net 0.0005 --lr_block 0.0005\
    --log_dir $LOGDIR --num_features $NUM_FEATURES --image_size 224\
 --dir_for_saving_images $IMGDIR --seed 1234\
 --save_epoch_interval $SAVE_INTERVAL --eval_epoch_interval $EVAL_INTERVAL --visualize_epoch_interval 2 --state_dict_dir_net $NETDIR\
  --extra_test_image_folder $TEST_SCHEMA_ONLY_FINE_MARVEL_PATH\
