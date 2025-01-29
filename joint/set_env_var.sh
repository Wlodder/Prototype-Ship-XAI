#!/bin/bash

export PACKAGE_PATH='/home/wlodder/Interpretability/Prototypes/This-looks-like-those_ProtoConcepts/'

# Dataset diretories
export MMARVEL_BASE_PATH="/media/wlodder/T9/Datasets/Experiments/XAI/Military_Marvel"
export MMARVEL_DATA_PATH=$MMARVEL_BASE_PATH/data
export TRAIN_MMARVEL_PATH=$MMARVEL_BASE_PATH/train/data
export TEST_MMARVEL_PATH=$MMARVEL_BASE_PATH/test/data
export PUSH_MMARVEL_PATH=$TRAIN_MMARVEL_PATH
export VIS_MMARVEL_PATH=$TRAIN_MMARVEL_PATH/vis/data

# Dataset diretories Finegrained
export FINE_MARVEL_BASE_PATH="/media/wlodder/T9/Datasets/Experiments/XAI/FineGrainedVesselRecognition"
export FINE_MARVEL_DATA_PATH=$FINE_MARVEL_BASE_PATH
export TRAIN_FINE_MARVEL_PATH=$FINE_MARVEL_BASE_PATH/train/data/class_dataset
export TEST_FINE_MARVEL_PATH=$FINE_MARVEL_BASE_PATH/test/data/class_dataset
export TEST_SCHEMA_FINE_MARVEL_PATH=$FINE_MARVEL_BASE_PATH/test_proj_schema
export TEST_SCHEMA_ONLY_FINE_MARVEL_PATH=$FINE_MARVEL_BASE_PATH/schematics
export PUSH_FINE_MARVEL_PATH=$TRAIN_FINE_MARVEL_PATH
export VIS_FINE_MARVEL_PATH=$TRAIN_FINE_MARVEL_PATH/vis/data

# Dataset directories Janes
export JANES_MARVEL_BASE_PATH="/media/wlodder/T9/Datasets/Experiments/XAI/FineGrainedVesselRecognition/janes"
export TRAIN_JANES_MARVEL_PATH=$JANES_MARVEL_BASE_PATH/train/data/jane_dataset
export TEST_JANES_MARVEL_PATH=$JANES_MARVEL_BASE_PATH/test/data/jane_dataset
export TEST_JANES_LABEL_MARVEL_PATH=$JANES_MARVEL_BASE_PATH/test/json
export PUSH_JANES_MARVEL_PATH=$TRAIN_JANES_MARVEL_PATH
export VIS_JANES_MARVEL_PATH=$TRAIN_JANES_MARVEL_PATH/vis/data

# Running directories
export BASE_RUNDIR='/home/wlodder/Interpretability/Prototypes/This-looks-like-those_ProtoConcepts/'
export TESNET_RUNDIR=$BASE_RUNDIR/TesNet-Concept_final/
export PROTO_POOL_RUNDIR=$BASE_RUNDIR/ProtoPool-Concept_final/
export PROTO_PNET_RUNDIR=$BASE_RUNDIR/ProtoPNet-Concept_final/
export PIPNET_RUNDIR=$BASE_RUNDIR/PIPNet/

# Results directories
export TESNET_RESULTS_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results
export TESNET_PROTO_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results

export PROTO_POOL_RESULTS_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results
export PROTO_POOL_PROTO_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results

export PROTO_PNET_RESULTS_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results
export PROTO_PNET_PROTO_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results

export PIPNET_RESULTS_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results
export PIPNET_PROTO_DIR=/media/wlodder/T9/Datasets/Experiments/XAI/proto_results
