
LIST=(net_trained_last net_trained_26)
#for f in $(find /media/wlodder/T9/Datasets/Experiments/XAI/proto_results/resnet50_10_30_400/log/checkpoints/ -type f -exec basename {} \;); 


for f in $LIST
do 
    echo $f;
    ../joint/settings/PIP_settings_Fine_Marvel_exploration.sh $f; 
done