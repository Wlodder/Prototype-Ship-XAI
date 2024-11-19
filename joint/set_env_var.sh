#!/bin/bash

export PACKAGE_PATH='/home/wlodder/Interpretability/Prototypes/This-looks-like-those_ProtoConcepts/'

# Dataset diretories
export MMARVEL_BASE_PATH="/media/wlodder/T9/Datasets/Experiments/XAI/Military_Marvel"
export MMARVEL_DATA_PATH=$MMARVEL_BASE_PATH/data
export TRAIN_MMARVEL_PATH=$MMARVEL_BASE_PATH/train/data
export TEST_MMARVEL_PATH=$MMARVEL_BASE_PATH/test/data
export PUSH_MMARVEL_PATH=$TRAIN_MMARVEL_PATH
export VIS_MMARVEL_PATH=$TRAIN_MMARVEL_PATH/vis/data

# Running directories
export BASE_RUNDIR='/home/wlodder/Interpretability/Prototypes/This-looks-like-those_ProtoConcepts/'
export TESNET_RUNDIR=$BASE_RUNDIR/TesNet-Concept_final/
export PROTO_POOL_RUNDIR=$BASE_RUNDIR/ProtoPool-Concept_final/
export PROTO_PNET_RUNDIR=$BASE_RUNDIR/ProtoPNet-Concept_final/

# Results directories
export TESNET_RESULTS_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results
export TESNET_PROTO_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results

export PROTO_POOL_RESULTS_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results
export PROTO_POOL_PROTO_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results

export PROTO_PNET_RESULTS_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results
export PROTO_PNET_PROTO_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results