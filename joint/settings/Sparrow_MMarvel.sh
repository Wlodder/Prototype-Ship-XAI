#!/bin/bash 

# Paths
TRAIN_PATH=$TRAIN_JANES_MARVEL_PATH
TEST_PATH=$TEST_JANES_MARVEL_PATH
PUSH_PATH=$PUSH_JANES_MARVEL_PATH

# Architecture and model
N_CLASSES=$(ls $TRAIN_PATH | wc -l)
MODEL="resnet50"
PROTOTYPE_ACTIVATION_FUNCTION="log"

# Data and paths
BATCH_SIZE=80

# Prototypes
if [ $N_CLASSES -lt 40 ]; then
    NUM_PROTOTYPES=$((N_CLASSES * 40))
else
    # 137 -> 137 * 25
    NUM_PROTOTYPES=$((N_CLASSES * 25))
fi
PROTO_DEPTH=512
NUM_DESCRIPTIVE=10
K=10

# Epoch and training
# warmup epochs must be less than the total number of epochs
WARMUP_EPOCHS=15
EPOCH=45

# Losses
CAPALL=True
TOPKLOSS=True
ONLYWARMUP=True
SEP=True
CAPCOEF=0.01
CAPL=4.5

DATASET_NAME=$(basename $TRAIN_PATH)
RESULTS_PATH=$SPARROW_RESULTS_DIR/Sparrow/${DATASET_NAME}/${PROTO_DEPTH}_${NUM_PROTOTYPES}_${PROTOTYPE_ACTIVATION_FUNCTION}


python $SPARROW_RUNDIR/main_ctrl_caps.py --num_classes $N_CLASSES --train_batch_size $BATCH_SIZE --test_batch_size $BATCH_SIZE \
   --proto_depth $PROTO_DEPTH --base_architecture $MODEL --num_warm_epochs $WARMUP_EPOCHS \
  --prototype_activation_function $PROTOTYPE_ACTIVATION_FUNCTION --num_train_epochs $EPOCH \
     --cap_width $CAPCOEF --model_dir $RESULTS_PATH --num_prototypes $NUM_PROTOTYPES 
      # --gpuid 0 
   #   --k $K \
