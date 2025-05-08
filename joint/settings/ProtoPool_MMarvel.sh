#!/bin/bash 

# Paths
TRAIN_PATH=$TRAIN_JANES_MARVEL_PATH
TEST_PATH=$TEST_JANES_MARVEL_PATH
PUSH_PATH=$PUSH_JANES_MARVEL_PATH

N_CLASSES=$(ls $TRAIN_PATH | wc -l)
MODEL="resnet50"
PROTOTYPE_ACTIVATION_FUNCTION="log"
BATCH_SIZE=80
NUM_PROTOTYPES=1500
PROTO_DEPTH=256
WARMUP_EPOCHS=10
NUM_DESCRIPTIVE=10
EPOCH=30
K=10
CAPALL=True
TOPKLOSS=True
ONLYWARMUP=True
SEP=True
CAPCOEF=0.003
CAPL=4.5


DATASET_NAME=$(basename $TRAIN_PATH)
RESULTS_PATH=$PROTO_POOL_RESULTS_DIR/ProtoPool/${DATASET_NAME}/${PROTO_DEPTH}_${NUM_PROTOTYPES}_${PROTOTYPE_ACTIVATION_FUNCTION}
OUTPUT_DIR=capwith_${CAPL}capcoef_${CAPCOEF}lr_upby100only_warmup_${ONLYWARMUP}sep_${SEP}topkclst_${TOPKLOSS}cap_all_${CAPALL}top_k_${K}control

mkdir -p $RESULTS_PATH

# echo Starting training
python $PROTO_POOL_RUNDIR/main_cap_final.py --num_classes $N_CLASSES --batch_size $BATCH_SIZE --num_descriptive $NUM_DESCRIPTIVE \
 --num_prototypes $NUM_PROTOTYPES ---use_scheduler \
 --arch $MODEL --pretrained --proto_depth $PROTO_DEPTH --warmup_time $WARMUP_EPOCHS --warmup \
  --prototype_activation_function $PROTOTYPE_ACTIVATION_FUNCTION --top_n_weight 0 --last_layer --use_thresh \
   --mixup_data --pp_ortho --pp_gumbel --gumbel_time 30 --data_train $TRAIN_PATH --data_push $PUSH_PATH \
    --data_test $TEST_PATH --gpuid 0 --epoch $EPOCH --lr 0.5e-3 --earlyStopping 12 --cap_start 0 --capl $CAPL \
     --capcoef $CAPCOEF --only_warmuptr $ONLYWARMUP --topk_loss $TOPKLOSS --k_top $K --sep $SEP \
     --results $RESULTS_PATH --proto_img_dir $RESULTS_PATH/img


EPOCH_SEARCH=$((EPOCH - 1))
MODEL_PATH=$(find $RESULTS_PATH -name *.pth | grep final | grep $EPOCH_SEARCH)
SAVE_DIR=$RESULTS_PATH/local_analysis/
VIS_PATH=$(dirname $(find $RESULTS_PATH -name *npy | grep $EPOCH_SEARCH))
echo Starting visualization for model $MODEL_PATH
echo $VIS_PATH
python $PROTO_POOL_RUNDIR/local_analysis_final.py --data_train $TRAIN_PATH --data_push $PUSH_PATH --data_test $TEST_PATH --model_dir $MODEL_PATH \
 --batch_size $BATCH_SIZE --num_descriptive $NUM_DESCRIPTIVE --num_prototypes $NUM_PROTOTYPES \
  --num_classes $N_CLASSES --arch $MODEL --gpuid 0 --pretrained --proto_depth $PROTO_DEPTH \
   --prototype_activation_function $PROTOTYPE_ACTIVATION_FUNCTION --last_layer --use_thresh --capl $CAPL \
   --save_analysis_dir $SAVE_DIR --target_img_dir $TEST_PATH --prototype_img_dir $VIS_PATH
