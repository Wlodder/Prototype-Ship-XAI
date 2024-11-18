#!/bin/bash 

N_CLASSES=8
MODEL="resnet50"
train_path="/media/wlodder/T9/Datasets/Experiments/XAI/Military_MARVEL/train_cropped_augmented/"
test_path=$train_path/vis_examples
push_path=$train_path/data
model_dir='./results/checkpoint/capwith_4.5capcoef_0.003lr_upby100only_warmup_Truesep_Truetopkclst_Truecap_all_Falsetop_k_10control/'
model_path=${model_dir}/model_final_epoch29.pth
save_dir='./local_analysis/save_dir/'
prototype_vis_dir='./results/img_proto/capwith_4.5capcoef_0.003lr_upby100only_warmup_Truesep_Truetopkclst_Truecap_all_Falsetop_k_10control/epoch-29'

python main_cap_final.py --num_classes $N_CLASSES --batch_size 80 --num_descriptive 10 --num_prototypes 195 ---use_scheduler \
 --arch resnet50 --pretrained --proto_depth 256 --warmup_time 10 --warmup \
  --prototype_activation_function log --top_n_weight 0 --last_layer --use_thresh \
   --mixup_data --pp_ortho --pp_gumbel --gumbel_time 30 --data_train $train_path --data_push $push_path \
    --data_test $test_path --gpuid 0 --epoch 30 --lr 0.5e-3 --earlyStopping 12 --cap_start 0 --capl 4.5 \
     --capcoef 3e-3 --only_warmuptr True --topk_loss True --k_top 10 --sep True

python local_analysis_final.py --data_train $train_path --data_push $push_path --data_test $test_path --model_dir $model_path \
 --batch_size 80 --num_descriptive 10 --num_prototypes 195 --num_classes $N_CLASSES --arch "resnet50" --gpuid 0 --pretrained \ 
  --proto_depth 256 --prototype_activation_function log --last_layer --use_thresh --capl 4.5 \
   --save_analysis_dir $save_dir --target_img_dir $test_path --prototype_img_dir $prototype_vis_dir
/ 