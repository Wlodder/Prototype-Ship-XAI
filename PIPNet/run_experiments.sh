#!/bin/bash

# for i in {3..6};
# do
# 	source ../joint/set_env_var_template.sh trial_5_${i}
# 	echo running $i
# 	../joint/settings/PIP_settings_Fine_Marvel_janes_cutmix.sh
# done

for i in 10 25 50 100;
do
	source ../joint/set_env_var_template.sh trial_5_2
	echo running $i
	../joint/settings/PIP_settings_Fine_Marvel_janes_cutmix.sh $i
done
