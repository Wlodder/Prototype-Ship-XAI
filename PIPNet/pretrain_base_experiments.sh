#!/bin/bash
# source ../joint/set_env_var_template.sh janes
# ../joint/settings/PIP_settings_Fine_Marvel_janes_cutmix_pretrain.sh 200

# This script is used to test pre-expert system for longtailed interpretable feature learning.



# We just want to use the maximum batch size as possible
batch_size=24

# Different sets of prototypes

source ../joint/set_env_var_template.sh janes janes
for dropout in 0.0 #0.05 0.15 # 0.3 0.5 0.7 0.9
do

	for num_proto in  100 200 400 600 800 
	do
		# ../joint/settings/PIP_settings_Fine_Marvel_janes_pure.sh $num_proto $batch_size $dropout
		for alpha in 1 2 3 6 8 
		do
			../joint/settings/PIP_settings_Fine_Marvel_janes_crp.sh $num_proto $batch_size $dropout $alpha
		done
	done
done