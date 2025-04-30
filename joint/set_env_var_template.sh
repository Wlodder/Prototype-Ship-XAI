#!/bin/bash

# Change-able base directories
DATA_DIRECTORY=$1
WEIGHT_DIRECTORY=$2
export DATA_DIRECTORY=$DATA_DIRECTORY
export WEIGHT_DIRECTORY=$WEIGHT_DIRECTORY
export BASE_RUNDIR='/home/wlodder/Interpretability/Prototypes/This-looks-like-those_ProtoConcepts/'
export BASE_RESULTS_DIR="/media/wlodder/Data/XAI/proto_results"
export DATA_BASE_DIR='/media/wlodder/Data/XAI/JaneOnlyFineGrainedVesselRecognition'
export PACKAGE_PATH=$BASE_RUNDIR

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
export JANES_EXPERIMENT_DIRECTORY=$WEIGHT_DIRECTORY
# /media/wlodder/Data/XAI/JaneOnlyFineGrainedVesselRecognition
export JANES_MARVEL_BASE_PATH="${DATA_BASE_DIR}/${DATA_DIRECTORY}"
# export JANES_MARVEL_BASE_PATH="/media/wlodder/T9/Datasets/Experiments/XAI/FineGrainedVesselRecognition/${DIRECTORY}"
export TRAIN_JANES_MARVEL_PATH=$JANES_MARVEL_BASE_PATH/train/data/jane_dataset
export TEST_JANES_MARVEL_PATH=$JANES_MARVEL_BASE_PATH/test/data/jane_dataset
export TEST_JANES_LABEL_MARVEL_PATH=$JANES_MARVEL_BASE_PATH/test/json
export PUSH_JANES_MARVEL_PATH=$TRAIN_JANES_MARVEL_PATH
export VIS_JANES_MARVEL_PATH=$TRAIN_JANES_MARVEL_PATH/vis/data
export SUGGESTION_JANES_MARVEL_PATH=$JANES_MARVEL_BASE_PATH/suggestions/data/jane_dataset
export SUGGESTION_JANES_LABEL_MARVEL_PATH=$JANES_MARVEL_BASE_PATH/suggestions/json


if [ ! -d $JANES_MARVEL_BASE_PATH ]; then 
    echo "Chosen directory does not exist"
fi

# Running directories
export TESNET_RUNDIR=$BASE_RUNDIR/TesNet-Concept_final/
export PROTO_POOL_RUNDIR=$BASE_RUNDIR/ProtoPool-Concept_final/
export PROTO_PNET_RUNDIR=$BASE_RUNDIR/ProtoPNet-Concept_final/
export PIPNET_RUNDIR=$BASE_RUNDIR/PIPNet/
export STPROTOPNET_RUNDIR=$BASE_RUNDIR/ST-ProtoPNet/full/
export SPARROW_RUNDIR=$BASE_RUNDIR/Sparrow/

# Results directories
export TESNET_RESULTS_DIR=$BASE_RESULTS_DIR
export TESNET_PROTO_DIR=$BASE_RESULTS_DIR

export PROTO_POOL_RESULTS_DIR=$BASE_RESULTS_DIR
export PROTO_POOL_PROTO_DIR=$BASE_RESULTS_DIR

export STPROTOPNET_RESULTS_DIR=$BASE_RESULTS_DIR
export STPROTOPNET_PROTO_DIR=$BASE_RESULTS_DIR

export PROTO_PNET_RESULTS_DIR=$BASE_RESULTS_DIR
export PROTO_PNET_PROTO_DIR=$BASE_RESULTS_DIR

export PIPNET_RESULTS_DIR=$BASE_RESULTS_DIR
export PIPNET_PROTO_DIR=$PIPNET_RESULTS_DIR

export SPARROW_RESULTS_DIR=$BASE_RESULTS_DIR
export SPARROW_PROTO_DIR=$BASE_RESULTS_DIR