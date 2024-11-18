import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size',default = 224)
    parser.add_argument('--prototype_shape',default = (2000, 64, 1, 1))
    parser.add_argument('--num_classes',default= 8)
    parser.add_argument('--prototype_activation_function',default= 'log')
    parser.add_argument('--add_on_layers_type',default= 'regular')

    parser.add_argument('--experiment_run',default= 'MMarvel')
    parser.add_argument('--base_architecture',default= 'vgg19')

    parser.add_argument('--train_batch_size',default= 80)
    parser.add_argument('--test_batch_size',default= 100)
    parser.add_argument('--train_push_batch_size',default=75)
    parser.add_argument('--joint_lr_step_size',default= 5)


    parser.add_argument('--k',default= 3)
    parser.add_argument('--cap_width',default= 8.05 )


    parser.add_argument('--num_train_epochs',default= 10)
    parser.add_argument('--num_warm_epochs',default= 5)

    parser.add_argument('--push_start',default= 10)
    parser.add_argument('--analysis_start',default= 0)
    return parser

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3,
                      'cap_width_l2': 1e-4}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.2,# by a lot 
    'l1': 1e-4,
    'orth': 5e-3, # by 10
    'sub_sep': -5e-5, #by 10 
    'cap_coef':3e-5
}

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3,
                       'cap_width_l2': 1e-5}