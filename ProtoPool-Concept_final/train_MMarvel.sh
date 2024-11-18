#!/bin/bash 

N_CLASSES=8
MODEL="resnet50"
train_path="/media/wlodder/T9/Datasets/Experiments/XAI/Military_MARVEL/train_cropped_augmented/data"
test_path=$train_path
push_path=$train_path

python main_cap_final.py --num_classes 196 --batch_size 80 --num_descriptive 10 --num_prototypes 195 \  
    ---use_scheduler --arch resnet50 --pretrained --proto_depth 256 --warmup_time 10 --warmup \ 
    --prototype_activation_function log --top_n_weight 0 --last_layer --use_thresh --mixup_data \ 
    --pp_ortho --pp_gumbel --gumbel_time 30 --data_train $train_path --data_push $push_path --data_test $test_path \ 
    --gpuid 0 --epoch 30 --lr 0.5e-3 --earlyStopping 12 --cap_start 0 --capl 4.5 --capcoef 3e-3 \
     --only_warmuptr True --topk_loss True --k_top 10 --sep True


