#!/bin/bash
# general settings
NUM_FEATURES=800
SEED=1234
FILENAME=$1
BATCH_SIZE=24
DROPOUT=0.0
BATCH_SIZE_PRETRAIN=32
NET="convnext_tiny_26"
EPOCHS_PRETRAIN=6
EPOCHS_TRAIN=30
SAVE_INTERVAL=2
EVAL_INTERVAL=10
VIS_INTERVAL=5
IMAGE_SIZE=256
K=15
BUFFER_SIZE=6
TYPE='folder'

SUBJECT_DIR=${NET}_${EPOCHS_PRETRAIN}_${EPOCHS_TRAIN}_${NUM_FEATURES}_${BATCH_SIZE}_${BATCH_SIZE_PRETRAIN}_${IMAGE_SIZE}/PIPNet_squared/${JANES_EXPERIMENT_DIRECTORY}/${K}_${BUFFER_SIZE}_${DROPOUT}_main

# Create log dir and image dir
LOGDIR=$PIPNET_PROTO_DIR/$SUBJECT_DIR/log
IMGDIR=$PIPNET_PROTO_DIR/$DATA_DIRECTORY/$SUBJECT_DIR/img_dir
CHECKPOINT=$LOGDIR/checkpoints/$FILENAME

mkdir -p $LOGDIR
mkdir -p $IMGDIR

export CUDA_VISIBLE_DEVICES=0
B=10
A=1
python $PIPNET_RUNDIR/merger.py --dataset "Janes_Military_Marvel" --net $NET --batch_size $BATCH_SIZE\
  --batch_size_pretrain $BATCH_SIZE_PRETRAIN --epochs $EPOCHS_TRAIN\
    --epochs_pretrain $EPOCHS_PRETRAIN --lr 0.005 --lr_net 0.0005 --lr_block 0.0005\
    --log_dir $LOGDIR --num_features $NUM_FEATURES --image_size $IMAGE_SIZE\
  --dir_for_saving_images $IMGDIR --seed 1234\
  --save_epoch_interval $SAVE_INTERVAL --eval_epoch_interval $EVAL_INTERVAL --visualize_epoch_interval $VIS_INTERVAL\
  --extra_test_image_folder $SUGGESTION_JANES_MARVEL_PATH --proto_dir $LOGDIR --num_workers 4\
  --buffer_type buffer:35 --k 45 --patch_size 32 --state_dict_dir_net $CHECKPOINT\
  --debug --prototype_indices 3 30 69 73\
  --apply_splitting --similarity_type feature\
  --visualize_results --splitting_scale 0.000005 --clustering_algorithm "hdbscan_kmeans" -a $A -b $B --output_path ${A}_${B}\
  --split_prototype 
  # --check_prototype_locations #--split_prototype #--analyze_all_prototypes 
  # --analyze_all_prototypes\