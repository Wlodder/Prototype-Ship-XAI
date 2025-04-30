
dir_choice="full"
source ./joint/set_env_template $dir_choice $dir_choice
# ./joint/settings/PIP_settings_Fine_Marvel_janes_cutmix_pretrains.sh 1200 24 0.0
./joint/settings/TesNet_MMarvel.sh
./joint/settings/ST_ProtoPNet_MMarvel.sh
./joint/settings/ProtoPool_MMarvel.sh
./joint/settings/ProtoPNet_MMarvel.sh
./joint/settings/Sparrow_MMarvel.sh
