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

# Prototypes % classes == 0
NUM_PROTOTYPES=75
PROTO_DEPTH=256
NUM_DESCRIPTIVE=10
K=10

# Epoch and training
# warmup epochs must be less than the total number of epochs
WARMUP_EPOCHS=15
EPOCH=20

# Losses
CAPALL=True
TOPKLOSS=True
ONLYWARMUP=True
SEP=True
CAPCOEF=0.01
CAPL=4.5


RESULTS_PATH=$TESNET_RESULTS_DIR/TesNet/${PROTO_DEPTH}_${NUM_PROTOTYPES}_${PROTOTYPE_ACTIVATION_FUNCTION}


python $TESNET_RUNDIR/main_cap.py --num_classes $N_CLASSES --train_batch_size $BATCH_SIZE --test_batch_size $BATCH_SIZE \
 --num_prototypes $NUM_PROTOTYPES --proto_depth $PROTO_DEPTH \
 --base_architecture $MODEL --num_warm_epochs $WARMUP_EPOCHS \
  --prototype_activation_function $PROTOTYPE_ACTIVATION_FUNCTION --num_train_epochs $EPOCH \
     --cap_width $CAPCOEF --k $K --model_dir $RESULTS_PATH --gpuid 0 

MODEL_PATH=$(find $RESULTS_PATH -name *.pth | grep final)
SAVE_DIR=$RESULTS_PATH/local_analysis/
VIS_PATH=$(dirname $(find $RESULTS_PATH -name *npy))
echo $VIS_PATH
python $TESNET_RUNDIR/local_analysis.py --data_train $TRAIN_PATH --data_push $PUSH_PATH --data_test $TEST_PATH --model_dir $MODEL_PATH \
 --batch_size $BATCH_SIZE --num_descriptive $NUM_DESCRIPTIVE --num_prototypes $NUM_PROTOTYPES \
  --num_classes $N_CLASSES --arch $MODEL --gpuid 0 --pretrained --proto_depth $PROTO_DEPTH \
   --prototype_activation_function $PROTOTYPE_ACTIVATION_FUNCTION --last_layer --use_thresh --capl $CAPL \
   --save_analysis_dir $SAVE_DIR --target_img_dir $TEST_PATH --prototype_img_dir $VIS_PATH
