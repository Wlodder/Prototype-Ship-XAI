#!/bin/bash 

# Architecture and model
N_CLASSES=8
MODEL="resnet50"
PROTOTYPE_ACTIVATION_FUNCTION="log"

# Data and paths
BATCH_SIZE=80

# Prototypes
NUM_PROTOTYPES=64
PROTO_DEPTH=256
NUM_DESCRIPTIVE=10
K=10

# Epoch and training
# warmup epochs must be less than the total number of epochs
WARMUP_EPOCHS=4
EPOCH=5

# Losses
CAPALL=True
TOPKLOSS=True
ONLYWARMUP=True
SEP=True
CAPCOEF=0.01
CAPL=4.5


TRAIN_PATH=$TRAIN_MMARVEL_PATH
TEST_PATH=$TEST_MMARVEL_PATH
PUSH_PATH=$PUSH_MMARVEL_PATH

RESULTS_PATH=$PROTO_PNET_RESULTS_DIR/ProtoPNet/${PROTO_DEPTH}_${NUM_PROTOTYPES}_${PROTOTYPE_ACTIVATION_FUNCTION}


python $PROTO_PNET_RUNDIR/main_ctrl_caps.py --num_classes $N_CLASSES --train_batch_size $BATCH_SIZE --test_batch_size $BATCH_SIZE \
   --proto_depth $PROTO_DEPTH --base_architecture $MODEL --num_warm_epochs $WARMUP_EPOCHS \
  --prototype_activation_function $PROTOTYPE_ACTIVATION_FUNCTION --num_train_epochs $EPOCH \
     --cap_width $CAPCOEF --model_dir $RESULTS_PATH --num_prototypes $NUM_PROTOTYPES 
      # --gpuid 0 
   #   --k $K \

# MODEL_PATH=$(find $RESULTS_PATH -name *.pth | grep final)
# SAVE_DIR=$RESULTS_PATH/local_analysis/
# VIS_PATH=$(dirname $(find $RESULTS_PATH -name *npy))
# echo $VIS_PATH
# python $TESNET_RUNDIR/local_analysis_final.py --data_train $TRAIN_PATH --data_push $PUSH_PATH --data_test $TEST_PATH --model_dir $MODEL_PATH \
#  --batch_size $BATCH_SIZE --num_descriptive $NUM_DESCRIPTIVE --num_prototypes $NUM_PROTOTYPES \
#   --num_classes $N_CLASSES --arch $MODEL --gpuid 0 --pretrained --proto_depth $PROTO_DEPTH \
#    --prototype_activation_function $PROTOTYPE_ACTIVATION_FUNCTION --last_layer --use_thresh --capl $CAPL \
#    --save_analysis_dir $SAVE_DIR --target_img_dir $TEST_PATH --prototype_img_dir $VIS_PATH
